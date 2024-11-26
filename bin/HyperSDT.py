# %%
import math
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor

import lib


# %%
class Tokenizer(nn.Module):
    """
    References:
    - FT-Transformer: https://github.com/Yura52/tabular-dl-revisiting-models/blob/main/bin/ft_transformer.py#L18
    - T2G_former: https://github.com/jyansir/t2g-former/blob/master/bin/t2g_former.py
    """
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape}')
        print(d_bias)
        # take [CLS] token into account
        # ----------------HN 2024-8-9---------------- #
        # self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.weight = nn.Parameter(Tensor(d_numerical, d_token))
        # ----------------HN 2024-8-9---------------- #
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x = x_num if x_cat is None else torch.cat([x_num, x_cat], dim=1)
        x = self.weight[None] * x_num[:, :, None]  # 注意这里x_num的维度已经改变

        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )

        if self.bias is not None:
            x = x + self.bias[None]  # 调整偏置的维度

        return x


class MultiheadAttention(nn.Module):
    def __init__(
        self, d: int, n_heads: int, dropout: float, initialization: str,
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: ty.Optional[nn.Linear],
        value_compression: ty.Optional[nn.Linear]
    ) -> Tensor:
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        #q-->[batch_size * self.n_heads, n_tokens_q, d_head]
        k = self._reshape(k)
        #k-->[batch_size * self.n_heads, n_tokens_k, d_head]
        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x



class SlideDecisionTree(nn.Module):
    def __init__(
        self,
        tree_type: str = 'SDT',
        activation: str = 'entmax'
        ):
        super().__init__()
        self.tree_type = tree_type
        self.activation_fn = lib.get_activation_fn(activation)
        self.sg = nn.ReLU()
        
    def forward(self, x, weight, thres=None) -> Tensor:
        # if self.tree_type == 'SDT':
        #     # print(self.activation_fn(weight/0.1))
        #     decision_layer = torch.einsum('bn, bnd->bd', weight, x)
        # elif self.tree_type == "DT":
        #     decision_layer = torch.einsum('bn, bnd->bd', weight, x) - thres
        # return self.activation_fn(decision_layer)
        weight = self.activation_fn(weight)
        decision_layer = torch.einsum('bn, bnd->bd', weight, x)
        # decision_layer = self.sg(decision_layer)
        return decision_layer


class DecisionTransformer(nn.Module):
    """Transformer.

    References:
    - FT-Transformer: https://github.com/Yura52/tabular-dl-revisiting-models/blob/main/bin/ft_transformer.py#L151
    - T2G-Former: https://github.com/jyansir/t2g-former/blob/master/bin/t2g_former.py
    """

    def __init__(
        self,
        *,
        # tokenizer
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        token_bias: bool,
        # transformer
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_ffn_factor: float,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        activation: str,
        prenormalization: bool,
        initialization: str,
        # linformer
        kv_compression: ty.Optional[float],
        kv_compression_sharing: ty.Optional[str],
        #
        d_out: int,

        sd_activation: str,
        device
    ) -> None:
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)

        super().__init__()

        self.tokenizer = Tokenizer(d_numerical, categories, d_token, token_bias)
        n_tokens = self.tokenizer.n_tokens
        # ----------------HN 2024-8-9---------------- #
        self.n_decision = n_layers
        self.d_token = d_token
        self.device = device

        self.sd_activation = sd_activation
        self.activation_weight = lib.get_activation_fn(sd_activation)
        # ----------------HN 2024-8-9---------------- #
        
        def make_kv_compression():
            assert kv_compression
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            if initialization == 'xavier':
                nn_init.xavier_uniform_(compression.weight)
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == 'layerwise'
            else None
        )

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        # self.encoder = nn.ModuleDict(
        #     {
        #         'attention': MultiheadAttention(
        #             d_token, n_heads, attention_dropout, initialization
        #         ),
        #         'linearx0': nn.Linear(
        #             d_token, d_hidden * (2 if activation.endswith('glu') else 1)
        #         ),
        #         'linearx1': nn.Linear(d_hidden, d_token),
        #         'norm1': make_normalization(),
        #     }
        # )
        self.encoder_layers = nn.ModuleList([])
        self.encoder_len = n_layers
        for i in range(self.encoder_len):
            encoder_layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linearx0': nn.Linear(
                        d_token, d_hidden * (2 if activation.endswith('glu') else 1)
                    ),
                    'linearx1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            self.encoder_layers.append(encoder_layer)


        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden * (2 if activation.endswith('glu') else 1)
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm2': make_normalization(),
                    'toparam': nn.Linear(d_token, n_tokens),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.layers.append(layer)

        self.sd = SlideDecisionTree(activation=sd_activation)
        self.activation = lib.get_activation_fn(activation)
        self.last_activation = lib.get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

        # self.weights = []

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x
    
    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x = self.tokenizer(x_num, x_cat)
        decisions = torch.zeros(x.shape[0], self.n_decision + 1, self.d_token).to(self.device)

        # encoder
        for layer_idx, layer in enumerate(self.layers):
            x_residual = self._start_residual(x, self.encoder_layers[layer_idx], 0)
            x_residual = self.encoder_layers[layer_idx]['attention'](
                x_residual,
                x_residual,
                *self._get_kv_compressions(self.encoder_layers[layer_idx]),
            )
            x = self._end_residual(x, x_residual, self.encoder_layers[layer_idx], 0)

            x_residual = self._start_residual(x, self.encoder_layers[layer_idx], 1)
            x_residual = self.encoder_layers[layer_idx]['linearx0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = self.encoder_layers[layer_idx]['linearx1'](x_residual)
            x = self._end_residual(x, x_residual, self.encoder_layers[layer_idx], 1)
        # self.weights = []
        # self.weights1 = []
        # self.decision_ = []
        # self.x_ = x_num[0]
        
        # decoder
        
            # decoder_layer
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            decisions_cp = decisions.clone()
            d_residual = self._start_residual(decisions_cp, layer, 0)
            d_residual = layer['attention'](
                d_residual,
                x,
                *self._get_kv_compressions(layer),
            )
            d_residual = self._end_residual(decisions_cp, d_residual, layer, 0)

            d_residual = self._start_residual(decisions_cp, layer, 2)
            d_residual = layer['linear0'](d_residual)
            d_residual = self.activation(d_residual)
            if self.ffn_dropout:
                d_residual = F.dropout(d_residual, self.ffn_dropout, self.training)
            d_residual = layer['linear1'](d_residual)
            decisions_cp = self._end_residual(decisions_cp, d_residual, layer, 2)
            

            decision = decisions_cp[:, 0, :]        

            decision = layer['toparam'](decision) 
            weight = decision.squeeze(1) 

            # self.weights.append(self.activation_weight(weight[0]/0.1))

            layer_decision = self.sd(x, weight)

            decisions[:, layer_idx+1] = layer_decision

            if self.last_normalization is not None:
                layer_decision = self.last_normalization(layer_decision)
            layer_decision = self.last_activation(layer_decision)
            layer_decision = self.head(layer_decision)
            layer_decision = layer_decision.squeeze(-1)
            # self.decision_.append(layer_decision[0])

        return layer_decision
