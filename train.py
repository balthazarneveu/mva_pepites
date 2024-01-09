import sys
import argparse
from typing import Optional
import torch
import logging
from pathlib import Path
import json
from tqdm import tqdm
from shared import (
    ROOT_DIR, OUTPUT_FOLDER_NAME,
    ID, NAME, NB_EPOCHS,
    TRAIN, VALIDATION, TEST,
)
from experiments import get_experiment_config, get_training_content
WANDB_AVAILABLE = False
try:
    WANDB_AVAILABLE = True
    import wandb
except ImportError:
    logging.warning("Could not import wandb. Disabling wandb.")
    pass


def get_parser(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("-e", "--exp", nargs="+", type=int, required=True, help="Experiment id")
    parser.add_argument("-o", "--output-dir", type=str, default=ROOT_DIR/OUTPUT_FOLDER_NAME, help="Output directory")
    parser.add_argument("-nowb", "--no-wandb", action="store_true", help="Disable weights and biases")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    return parser


def training_loop(
    model, optimizer, dl_dict: dict, config: dict,
    device: str = "cuda", wandb_flag: bool = False,
    output_dir: Path = None
):
    for n_epoch in tqdm(range(config[NB_EPOCHS])):
        current_loss = {TRAIN: 0., VALIDATION: 0., TEST: 0.}
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
                current_loss[phase] += loss.item()
            current_loss[phase] /= (len(dl_dict[phase]))
        for phase in [VALIDATION, TEST]:
            print(f"{phase}: Epoch {n_epoch} - Loss: {current_loss[phase]:.3e}")
        if output_dir is not None:
            with open(output_dir/f"metrics_{n_epoch}.json", "w") as f:
                json.dump(current_loss, f)
        if wandb_flag:
            wandb.log(current_loss)
    if output_dir is not None:
        torch.save(model.cpu().state_dict(), output_dir/"last_model.pt")
    return model


def train(config: dict, output_dir: Path, device: str = "cuda", wandb_flag: bool = False):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Training experiment {config[ID]} on device {device}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir/"config.json", "w") as f:
        json.dump(config, f)
    model, optimizer, dl_dict = get_training_content(config, device=device)
    model.to(device)
    if wandb_flag:
        import wandb
        wandb.init(
            project="mva-pepites",
            name=config[NAME],
            tags=["debug"],
            config=config
        )
    model = training_loop(model, optimizer, dl_dict, config, device=device,
                          wandb_flag=wandb_flag, output_dir=output_dir)

    if wandb_flag:
        wandb.finish()


def train_main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    if not WANDB_AVAILABLE:
        args.no_wandb = True
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    for exp in args.exp:
        config = get_experiment_config(exp)
        print(config)
        output_dir = Path(args.output_dir)/config[NAME]
        logging.info(f"Training experiment {config[ID]} on device {device}...")
        train(config, device=device, output_dir=output_dir, wandb_flag=not args.no_wandb)


if __name__ == "__main__":
    train_main(sys.argv[1:])
