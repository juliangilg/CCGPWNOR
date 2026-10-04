"""Command line entry point.

Example:
    python main.py --dataset rsna --data_dir ./data --epochs 1 --seeds 0
"""

import argparse

from mil_gp import CONFIGS, prepare_dataset, load_datasets, make_loaders, run_experiments
from mil_gp.data import DEFAULT_DATA_DIR


def parse_args():
    p = argparse.ArgumentParser(description="MIL-GP with weighted Noisy-OR")
    p.add_argument("--dataset", required=True, choices=list(CONFIGS))
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                   help="Where the compressed files are stored/extracted (default: /content)")
    p.add_argument("--save_dir", default=None, help="Results folder (default: results/<dataset>)")
    p.add_argument("--features", default=None, help="Feature extractor (default: UNI / resnet50)")
    p.add_argument("--remove_archives", action="store_true", help="Delete .tar.gz after extraction")
    # Optional overrides of the default configuration.
    p.add_argument("--mode", choices=["independent", "correlated"])
    p.add_argument("--inducing", type=int, nargs="+", help="Number(s) of inducing points M")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--device", default=None, help="'cpu' or 'cuda' (default: auto)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = CONFIGS[args.dataset]
    model_cfg, train_cfg = dict(cfg["model"]), dict(cfg["train"])
    if args.features == "resnet18":
        model_cfg["input_dim"] = 512  # other extractors: set input_dim accordingly

    # Apply command line overrides.
    if args.mode: model_cfg["mode"] = args.mode
    if args.inducing: train_cfg["inducing_values"] = args.inducing
    if args.seeds: train_cfg["seeds"] = args.seeds
    if args.epochs: train_cfg["epochs"] = args.epochs
    if args.lr: train_cfg["lr"] = args.lr
    batch_size = args.batch_size or cfg["batch_size"]

    root = prepare_dataset(args.dataset, args.data_dir, args.features, args.remove_archives)
    train, val, test = load_datasets(args.dataset, root, args.features,
                                     load_at_init=cfg["load_at_init"])
    loaders = make_loaders(train, val, test, batch_size=batch_size)

    run_experiments(*loaders, save_dir=args.save_dir or f"results/{args.dataset}",
                    model_config=model_cfg, device=args.device, **train_cfg)


if __name__ == "__main__":
    main()
