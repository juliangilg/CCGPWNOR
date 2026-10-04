"""Gaussian-process MIL with weighted Noisy-OR aggregation (torchmil datasets)."""

from .data import DATASETS, prepare_dataset, load_datasets, make_loaders
from .model import MILGP, WeightedNoisyORLikelihood
from .train_eval import MILTrainer, binary_metrics, select_threshold
from .experiments import CONFIGS, run_experiments, summarize_results
