from shared import (
    ID, NAME, NB_EPOCHS, DATALOADER, BATCH_SIZE,
    TRAIN, VALIDATION, TEST,
    ARCHITECTURE, MODEL,
    N_PARAMS,
    OPTIMIZER, LR, PARAMS
)
from model import ConvModel
import torch
from data_loader import get_dataloaders
from typing import Tuple


def get_experiment_config(exp: int) -> dict:
    config = {
        ID: exp,
        NAME: f"{exp:04d}",
        NB_EPOCHS: 5
    }
    config[DATALOADER] = {
        BATCH_SIZE: {
            TRAIN: 32,
            VALIDATION: 32,
            TEST: 32
        }
    }
    config[OPTIMIZER] = {
        NAME: "Adam",
        PARAMS: {
            LR: 1e-3
        }
    }
    if exp == 0:
        config[MODEL] = {
            ARCHITECTURE: dict(
                conv_feats=8,
                kernel_size=3
            ),
            NAME: "ConvModel"
        }
    elif exp == 1 or exp == 2:
        config[MODEL] = {
            ARCHITECTURE: dict(
                conv_feats=16,
                kernel_size=5
            ),
            NAME: "ConvModel"
        }
        if exp == 2:
            config[NB_EPOCHS] = 100
    return config


def get_training_content(config: dict, device="cuda") -> Tuple[ConvModel, torch.optim.Optimizer, dict]:
    model = ConvModel(1, **config[MODEL][ARCHITECTURE])
    assert config[MODEL][NAME] == ConvModel.__name__
    config[MODEL][N_PARAMS] = model.count_parameters()
    optimizer = torch.optim.Adam(model.parameters(), **config[OPTIMIZER][PARAMS])
    dl_dict = get_dataloaders(config, factor=1, device=device)
    return model, optimizer, dl_dict


if __name__ == "__main__":
    config = get_experiment_config(0)
    print(config)
    model, optimizer, dl_dict = get_training_content(config)
    print(len(dl_dict[TRAIN].dataset))
