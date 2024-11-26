# HyperSDT
This is the implementation and experiment results of the paper "HyperSDT: HyperNetwork Slide Decision Tree for
Interpretable Tabular Learning".

The implementation of HyperSDT in the original paper is `bin/HyperSDT.py`.

## How to test your model

You can test your models by adding them to `bin` directory and `bin/__init__.py`. Keep the same API we used in other models, and write your own evaluation script (`run_HyperSDT.py` as a reference).

## Datasets

*LICENSE*: by downloading our dataset you accept licenses of all its components. We do not impose any new restrictions in addition to those licenses. You can find the list of sources in the section "References" of our paper.

1. Download the data from this [link](https://huggingface.co/datasets/Sinario/hypersdt/tree/main)(~250M)
2. Unpack the .zip file to the root directory of the project: `unzip data.zip -d ./` (`mkdir data` if not exist)

## Acknowledgement

We sincerely appreciate the benchmark provided by Yura52‘s [work1](https://github.com/Yura52/tabular-dl-revisiting-models) and [work2](https://github.com/yandex-research/tabular-dl-num-embeddings) for fair comparison and script implementation.
