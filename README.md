# MIL-GP-WNOR: Gaussian Processes for Multiple Instance Learning with Weighted Noisy-OR

Sparse variational multi-output Gaussian process for Multiple Instance Learning (MIL) with
uncertainty estimation. Each instance gets two latent functions:

- `f_p` → instance score `p_ij = sigmoid(f_p)`
- `f_w` → instance weight `w_ij = softmax_j(f_w)` (normalised within the bag)

Instances are aggregated with a **weighted Noisy-OR**:

```
P(Y_i = 1) = 1 - Π_j (1 - p_ij) ** w_ij
```

The two outputs are modelled either as **independent** GPs or as **correlated** GPs
(linear model of coregionalisation, LMC). The model is trained by maximising the ELBO, using natural
gradients (NGD) for the variational parameters and Adam for everything else. It reports
predictive means and variances at bag and instance level.

## Datasets

The three benchmarks are downloaded from the [torchmil](https://huggingface.co/torchmil) Hugging Face
hub (pre-extracted features):

| Key          | Dataset                | Features (default) | Instance labels |
|--------------|------------------------|--------------------|-----------------|
| `camelyon16` | CAMELYON16 (WSI)       | UNI (1024-d)       | patch labels    |
| `panda`      | PANDA (WSI)            | UNI (1024-d)       | patch labels    |
| `rsna`       | RSNA-ICH (CT scans)    | ResNet50 (2048-d)  | slice labels    |

> Camelyon16 UNI features are about 13 GB compressed. RSNA is about 280 MB.

## Repository structure

```
mil-gp-wnor/
├── mil_gp/
│   ├── data.py          # download / extract / load datasets and dataloaders
│   ├── model.py         # MILGP model and weighted Noisy-OR likelihood
│   ├── train_eval.py    # MILTrainer (training, prediction, evaluation) and metrics
│   └── experiments.py   # default configs per dataset, experiment loop, summaries
├── main.py              # command line entry point
├── demo.ipynb           # end-to-end demo (Colab friendly)
└── requirements.txt
```

## Installation

```bash
git clone <this-repo-url>
cd mil-gp-wnor
pip install -r requirements.txt
```

## Usage

```bash
# Default settings of the paper experiments for each dataset
python main.py --dataset rsna
python main.py --dataset panda
python main.py --dataset camelyon16

# Choose where the compressed files are stored (default: /content, as in Google Colab)
python main.py --dataset rsna --data_dir ./data

# Quick run: 1 epoch, 1 seed, on CPU
python main.py --dataset rsna --data_dir ./data --epochs 1 --seeds 0 --device cpu
```

Main options (they override the defaults in `mil_gp/experiments.py`):

| Option | Description |
|---|---|
| `--data_dir` | Folder for the `.tar.gz` files and the extracted dataset (default `/content`) |
| `--save_dir` | Results folder (default `results/<dataset>`) |
| `--mode` | `independent` or `correlated` |
| `--inducing` | One or more numbers of inducing points, for example `--inducing 50 100 150` |
| `--seeds` | One or more random seeds (one run per seed) |
| `--epochs`, `--lr`, `--batch_size`, `--device` | Training settings |
| `--remove_archives` | Delete the `.tar.gz` files after extraction |

### Data layout

The data is downloaded to `--data_dir` and extracted as torchmil expects:

```
<data_dir>/                       <data_dir>/
├── splits.csv                    ├── splits.csv  -> moved to root/
└── patches_512/                  └── root/            (RSNA)
    ├── features/features_UNI/        ├── features/features_resnet50/
    ├── labels/                       ├── labels/
    ├── patch_labels/                 ├── slice_labels/
    └── coords/                       └── splits.csv
   (Camelyon16, PANDA)
```

For RSNA, `splits.csv` is downloaded outside `root/` and is moved into it automatically.
Bags with a missing file in any folder are removed before loading.

### Python API

```python
from mil_gp import CONFIGS, prepare_dataset, load_datasets, make_loaders, run_experiments

cfg = CONFIGS["rsna"]
root = prepare_dataset("rsna", data_dir="./data")
loaders = make_loaders(*load_datasets("rsna", root), batch_size=cfg["batch_size"])
results, history, tables = run_experiments(
    *loaders, save_dir="results/rsna", model_config=cfg["model"], **cfg["train"]
)
```

## Outputs

Each run writes to `save_dir`:

- `best_<mode>_M<M>_seed<seed>.pth`: best checkpoint (by validation AUC) with the thresholds
- `training_history.csv`: per-epoch training ELBO and validation AUC, NLPD and Brier score
- `test_metrics.csv`: test metrics per run
- `summary_<level>.csv`, `metrics_tables.tex`: mean ± std across seeds

Reported metrics (bag level and, when instance labels are used, instance level): Accuracy, F1, AUC,
NLPD, Brier, ECE, AURC (entropy) and AURC (predictive variance). Decision thresholds are selected on
the validation split (G-mean on the ROC curve). The validation split is a stratified 15 % of the
official training split.


