# BioVLM
BioVLM: Routing Prompts, Not Parameters, for Cross-Modality Generalization in Biomedical VLMs (ACL Findings 2026)


## How to install

### Create your environment:

```bash
$ conda create --name biovlm python=3.8
$ conda activate biovlm
$ git clone https://github.com/mainaksingha01/BioVLM
$ cd BioVLM
$ bash setup_env.sh

Our code uses [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch.git) codebase for dataset and training.

```

## Datasets

To download MedMNIST datasets, do
```bash
$ python download.py
```

## Code Instructions
 - [GDrive](https://drive.google.com/drive/folders/154fbxLT7lk5T_fTxtyQL9LBoybVj0qCo?usp=sharing) folder contains the data splits of the datasets. Put these files inside each of the data folders.
 - Clone the [dassl](https://github.com/KaiyangZhou/Dassl.pytorch/tree/master/dassl) folder inside this repo.
 - Replace the `dassl/engine/trainer.py` file with the modified [trainer](https://github.com/mainaksingha01/APPLeNet/blob/master/dassl/engine/trainer.py) file.

```shell
$ cd scripts
$ bash base2new_train.sh dataset
$ bash base2new_test.sh dataset
$ bash crossdata_train.sh dataset
$ bash crossdata_test.sh train_dataset test_dataset
$ bash fewshot_train.sh dataset
$ bash fewshot_test.sh dataset
```


## Bibtex

Please cite the paper if you use our work . Thanks.

```
@inproceedings{singha2026biovlm,
  title={BioVLM: Routing Prompts, Not Parameters, for Cross-Modality Generalization in Biomedical VLMs},
  author={Singha, Mainak and Gupta, Tanisha and Jha, Ankit and Khan, Muhammad Haris and Ghosh, Sayantani and Banerjee, Biplab},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  pages={40986--41005},
  year={2026}
}
```

## Acknowledgements

Thanks to the authors of [CoOp](https://github.com/KaiyangZhou/CoOp) as our code is mainly based on this repository.
