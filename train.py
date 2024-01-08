import sys
import argparse
from typing import Optional
import torch
import logging
from pathlib import Path
import json
from tqdm import tqdm
from shared import (
    ROOT_DIR,
    ID, NAME, NB_EPOCHS,
    TRAIN, VALIDATION, TEST,
)
from experiments import get_experiment_config, get_training_content


def get_parser(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("-e", "--exp", nargs="+", type=int, required=True, help="Experiment id")
    parser.add_argument("-o", "--output-dir", type=str, default=ROOT_DIR/"__output", help="Output directory")
    parser.add_argument("-nowb", "--no-wandb", action="store_true", help="Disable weights and biases")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    return parser


def training_loop(model, optimizer, dl_dict: dict, config: dict, device: str = "cuda"):
    for n_epoch in tqdm(range(config[NB_EPOCHS])):
        for phase in [TRAIN, VALIDATION, TEST]:
            if phase == TRAIN:
                model.train()
            else:
                model.eval()
            for x, y in tqdm(dl_dict[phase], desc=f"{phase} - Epoch {n_epoch}"):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == TRAIN):
                    y_pred = model(x)
                    loss = torch.nn.functional.mse_loss(y_pred, y)
                    if phase == TRAIN:
                        loss.backward()
                        optimizer.step()
            if phase == VALIDATION:
                print(f"{phase}: Epoch {n_epoch} - Loss: {loss.item():.3f}")


def train(config: dict, output_dir: Path, device: str = "cuda", wandb_flag: bool = False):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Training experiment {config[ID]} on device {device}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir/"config.json", "w") as f:
        json.dump(config, f)
    if wandb_flag:
        import wandb
        wandb.init(project="mva-pepites", config=config)
    model, optimizer, dl_dict = get_training_content(config)
    model.to(device)
    training_loop(model, optimizer, dl_dict, config, device=device)
    if wandb_flag:
        wandb.finish()


def train_main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(args.exp)
    for exp in args.exp:
        config = get_experiment_config(exp)
        print(config)
        output_dir = Path(args.output_dir)/config[NAME]
        logging.info(f"Training experiment {config[ID]} on device {device}...")
        train(config, device=device, output_dir=output_dir)


if __name__ == "__main__":
    train_main(sys.argv[1:])
