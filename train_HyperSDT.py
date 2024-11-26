import os
import math
import time
import json
import random
import argparse
import numpy as np
from pathlib import Path
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from category_encoders import CatBoostEncoder

import optuna

from bin import DecisionTransformer
from lib import Transformations, build_dataset, prepare_tensors, DATA, make_optimizer

import warnings
warnings.filterwarnings("ignore")

DATASETS = [
    'gesture', 'churn', 'eye', 'california', 
    'house', 'adult', 'otto', 'helena', 'jannis', 
    'higgs-small', 'fb-comments', 'year'
]

IMPLEMENTED_MODELS = [DecisionTransformer]

AUC_FOR_BINCLASS = False # if True use AUC metric for binclass task

def get_training_args():
    MODEL_CARDS = [x.__name__ for x in IMPLEMENTED_MODELS]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='configs/')
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--normalization", type=str, default='quantile')
    parser.add_argument("--model", type=str, required=True, choices=MODEL_CARDS)
    parser.add_argument("--seed", type=int, default=42, help='default tune seed is 42')
    parser.add_argument("--early_stop", type=int, default=16, help='default early stop epoch is 16 for DNN tuning')
    args = parser.parse_args()
    
    args.output = Path(args.output) / f'{args.dataset}/{args.model}'
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    
    return args

def seed_everything(seed=42):
    '''
    Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.
    '''
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



"""args"""
device = torch.device('cuda')
args = get_training_args()
seed_everything(args.seed)

"""Datasets and Dataloaders"""

dataset_name = args.dataset
T_cache = True
normalization = args.normalization if args.normalization != '__none__' else None
transformation = Transformations(normalization=normalization)
dataset = build_dataset(DATA / dataset_name, transformation, T_cache)

if AUC_FOR_BINCLASS and dataset.is_binclass:
    metric = 'roc_auc' # AUC (binclass)
else:
    metric = 'score' # RMSE (regression) or ACC (classification)

if dataset.X_num['train'].dtype == np.float64:
    dataset.X_num = {k: v.astype(np.float32) for k, v in dataset.X_num.items()}
# convert categorical features to numerical features with CatBoostEncoder
if dataset.X_cat is not None:
    cardinalities = dataset.get_category_sizes('train')
    enc = CatBoostEncoder(
        cols=list(range(len(cardinalities))), 
        return_df=False
    ).fit(dataset.X_cat['train'], dataset.y['train'])
    for k in ['train', 'val', 'test']:
        # 1: directly regard catgorical features as numerical
        dataset.X_num[k] = np.concatenate([enc.transform(dataset.X_cat[k]).astype(np.float32), dataset.X_num[k]], axis=1)



d_out = dataset.n_classes or 1
X_num, X_cat, ys = prepare_tensors(dataset, device=device)

if dataset.task_type.value == 'regression':
    y_std = ys['train'].std().item()

batch_size_dict = {
    'churn': 128, 'eye': 128, 'gesture': 128, 'california': 256, 'house': 256, 'adult': 256 , 
    'higgs-small': 512, 'helena': 512, 'jannis': 512, 'otto': 512, 'fb-comments': 512,
    'covtype': 1024, 'year': 1024, 'santander': 1024, 'microsoft': 1024, 'yahoo': 256}
val_batch_size = 1024 if args.dataset in ['santander', 'year', 'microsoft'] else 256 if args.dataset in ['yahoo'] else 8192
if args.dataset == 'epsilon':
    batch_size = 16 if args.dataset == 'epsilon' else 128 if args.dataset == 'yahoo' else 256
elif args.dataset not in batch_size_dict:
    if dataset.n_features <= 32:
        batch_size = 512
        val_batch_size = 8192
    elif dataset.n_features <= 100:
        batch_size = 128
        val_batch_size = 512
    elif dataset.n_features <= 1000:
        batch_size = 32
        val_batch_size = 64
    else:
        batch_size = 16
        val_batch_size = 16
else:
    batch_size = batch_size_dict[args.dataset]

num_workers = 0
data_list = [X_num, ys] if X_cat is None else [X_num, X_cat, ys]
train_dataset = TensorDataset(*(d['train'] for d in data_list))
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)
val_dataset = TensorDataset(*(d['val'] for d in data_list))
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=num_workers
)
test_dataset = TensorDataset(*(d['test'] for d in data_list))
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=num_workers
)
dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

"""Models"""
model_cls = eval(args.model)
n_num_features = dataset.n_num_features
cardinalities = dataset.get_category_sizes('train')
n_categories = len(cardinalities)
cardinalities = None if n_categories == 0 else cardinalities

"""Loss Function"""
loss_fn = (
    F.binary_cross_entropy_with_logits
    if dataset.is_binclass
    else F.cross_entropy
    if dataset.is_multiclass
    else F.mse_loss
)

"""utils function"""
def apply_model(model, x_num, x_cat=None):
    if any(issubclass(eval(args.model), x) for x in IMPLEMENTED_MODELS):
        return model(x_num, x_cat)
    else:
        raise NotImplementedError

@torch.inference_mode()
def evaluate(model, parts):
    model.eval()
    predictions = {}
    for part in parts:
        assert part in ['train', 'val', 'test']
        predictions[part] = []
        for batch in dataloaders[part]:
            x_num, x_cat, y = (
                (batch[0], None, batch[1])
                if len(batch) == 2
                else batch
            )
            predictions[part].append(apply_model(model, x_num, x_cat))
        predictions[part] = torch.cat(predictions[part]).cpu().numpy()
    prediction_type = None if dataset.is_regression else 'logits'
    return dataset.calculate_metrics(predictions, prediction_type)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

running_time = 0.
def train(model, optimizer):
    """Training"""
    n_epochs = 10000
    best_score = -np.inf
    no_improvement = 0
    EARLY_STOP = args.early_stop

    global running_time
    start = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        for iteration, batch in enumerate(train_loader):
            x_num, x_cat, y = (
                (batch[0], None, batch[1])
                if len(batch) == 2
                else batch
            )
            optimizer.zero_grad()
            y_tilda = apply_model(model, x_num, x_cat)
            # print(y_tilda.shape)
            # print(y.shape)
            loss = loss_fn(y_tilda, y)
            loss.backward()
            optimizer.step()

        scores = evaluate(model, ['val'])
        val_score = scores['val'][metric]
        print(val_score)
        if val_score > best_score:
            best_score = val_score
            print(' <<< BEST VALIDATION EPOCH')
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement == EARLY_STOP:
            break
    running_time += time.time() - start
    return best_score

# claim large datasets
large_datasets = ['year', 'covtype', 'microsoft']

model_param_spaces = {
    "DecisionTransformer": {
        "activation": "reglu",
        "attention_dropout": (0, 0.5, 'uniform'),
        "d_ffn_factor": (
            (2/3, 8/3, 'uniform')
            if args.dataset not in large_datasets
            else 4/3
        ),
        "d_token": (64,512,'int'),
        "ffn_dropout": (0, 0.5, 'uniform'),
        "initialization": "kaiming",
        "n_heads": (8, 16, 32, 64, 'int'),
        "n_layers": (
            (1, 4, 'int')
            if args.dataset not in large_datasets
            else (1,2,'int')
        ),
        "prenormalization": True,
        "residual_dropout": (0, 0.5, 'uniform'),
        "sd_activation": "entmax"
    }
}
d_embedding_dicts = {
    'DCNv2': (64,512,'int'),
    'MLP': (64,512,'int'),
    "NODE": 256,
    'ResNet': (64,512,'int'),
    'SNN': (64,512,'int'),
}
if cardinalities is not None:
    if args.model in d_embedding_dicts:
        model_param_spaces[args.model]['d_embedding'] = d_embedding_dicts[args.model]
training_param_spaces = {
    'DecisionTransformer': {
        'lr': (1e-5, 1e-3, 'loguniform'),
        'weight_decay': (1e-6, 1e-3, 'loguniform'),
        'optimizer': 'adamw'
    }
}

def needs_wd(name):
    return all(x not in name for x in ['tokenizer', '.norm', '.bias'])


def get_model_training_params(trial):
    model_args = model_param_spaces[args.model]
    training_args = {
        'batch_size': batch_size,
        'eval_batch_size': val_batch_size,
        **training_param_spaces[args.model],
    }
    model_params = {}
    training_params = {}
    for param, value in model_args.items():
        if isinstance(value, tuple):
            suggest_type = value[-1]
            if suggest_type != 'categorical':
                model_params[param] = eval(f'trial.suggest_{suggest_type}')(param, *value[:-1])
            else:
                model_params[param] = trial.suggest_categorical(param, choices=value[0])
        else:
            model_params[param] = value
    for param, value in training_args.items():
        if isinstance(value, tuple):
            suggest_type = value[-1]
            if suggest_type != 'categorical':
                training_params[param] = eval(f'trial.suggest_{suggest_type}')(param, *value[:-1])
            else:
                training_params[param] = trial.suggest_categorical(param, choices=value[0])
        else:
            training_params[param] = value
    return model_params, training_params

def objective(trial):
    cfg_model, cfg_training = get_model_training_params(trial)
    """set default"""
    cfg_model.setdefault('kv_compression', None)
    cfg_model.setdefault('kv_compression_sharing', None)
    cfg_model.setdefault('token_bias', True)

    """prepare model arguments"""
    if args.model in ['DecisionTransformer']:
        cfg_model = {
            'd_numerical': n_num_features,
            'd_out': d_out,
            'categories': cardinalities,
            'device': device,
            **cfg_model
        }
    model = model_cls(**cfg_model).to(device)

    """Optimizers"""
    if args.model in ['DecisionTransformer']:
        for x in ['tokenizer', '.norm', '.bias']:
            assert any(x in a for a in (b[0] for b in model.named_parameters()))
        parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
        parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
        optimizer = make_optimizer(
            cfg_training['optimizer'],
            (
                [
                    {'params': parameters_with_wd},
                    {'params': parameters_without_wd, 'weight_decay': 0.0},
                ]
            ),
            cfg_training['lr'],
            cfg_training['weight_decay'],
        )

    if torch.cuda.device_count() > 1:  # type: ignore[code]
        print('Using nn.DataParallel')
        model = nn.DataParallel(model)
    
    best_val_score = train(model, optimizer)
    return best_val_score


cfg_model = model_param_spaces[args.model]
const_params = {
    p: v for p, v in cfg_model.items()
    if not isinstance(v, tuple)
}
cfg_training = training_param_spaces[args.model]
const_training_params = {
    p: v for p, v in cfg_training.items()
    if not isinstance(v, tuple)
}
const_training_params['patience'] = args.early_stop
cfg_file = f'{args.output}/cfg-tmp.json'
def save_per_iter(study, trial):
    saved_model_cfg = {**const_params}
    saved_training_cfg = {**const_training_params}
    for k in cfg_model:
        if k not in saved_model_cfg:
            saved_model_cfg[k] = study.best_trial.params.get(k)
    for k in cfg_training:
        if k not in saved_training_cfg:
            saved_training_cfg[k] = study.best_trial.params.get(k)
    saved_training_cfg = {
        'batch_size': batch_size,
        'eval_batch_size': val_batch_size,
        **saved_training_cfg
    }
    hyperparams = {
        'time': running_time,
        'metric': metric,
        'eval_score': study.best_trial.value,
        'n_trial': study.best_trial.number,
        'dataset': args.dataset,
        'normalization': args.normalization,
        'model': saved_model_cfg,
        'training': saved_training_cfg,
    }
    with open(cfg_file, 'w') as f:
        json.dump(hyperparams, f, indent=4, ensure_ascii=False)

iterations = 50 if args.dataset in large_datasets else 100

# iterations = 30 # DEBUG
study = optuna.create_study(direction="maximize")
study.optimize(func=objective, n_trials=iterations, callbacks=[save_per_iter])


cfg_file = f'{args.output}/cfg.json'
for k in cfg_model:
    if k not in const_params:
        const_params[k] = study.best_params.get(k)
for k in cfg_training:
    if k not in const_training_params:
        const_training_params[k] = study.best_params.get(k)

const_training_params = {
    'batch_size': batch_size,
    'eval_batch_size': val_batch_size,
    **const_training_params
}

hyperparams = {
    'time': running_time,
    'metric': metric,
    'eval_score': study.best_value,
    'n_trial': study.best_trial.number,
    'dataset': args.dataset,
    'normalization': args.normalization,
    'model': const_params,
    'training': const_training_params,
}
with open(cfg_file, 'w') as f:
    json.dump(hyperparams, f, indent=4, ensure_ascii=False)