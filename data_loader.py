import torch
from torch.utils.data import DataLoader, Dataset
from shared import TRAIN, VALIDATION, TEST, DATALOADER, BATCH_SIZE
from typing import List


class WavesDataloader(Dataset):
    def __init__(
        self,
        length=64,
        n_samples=1000,
        device: str = "cuda",
        freeze: bool = False,
        freq_range: List[float] = [0, 10.],
        seed: int = 42,
    ):
        self.freeze = freeze
        self.device = device
        self.n_samples = n_samples
        if self.freeze:
            torch.manual_seed(seed)
            self.freqs = torch.linspace(freq_range[0], freq_range[1], n_samples, device=device)
            self.phases = torch.rand(n_samples, device=device)*2*torch.pi
        else:
            self.freq_range = torch.Tensor(freq_range).to(device)
        self.x = torch.linspace(0., 1., length, device=device)

    def __getitem__(self, index):
        if self.freeze:
            freq = self.freqs[index]
            phase = self.phases[index]
        else:
            freq = self.freq_range[0] + (self.freq_range[1]-self.freq_range[0])*torch.rand(1, device=self.device)
            phase = torch.rand(1, device=self.device)*2*torch.pi
        sig = torch.cos(freq*self.x+phase)
        return sig.unsqueeze(-2), freq.squeeze()

    def __len__(self):
        return self.n_samples


def get_dataloaders(config: dict, factor=1, device: str = "cuda"):
    dl_train = WavesDataloader(n_samples=800*factor, freeze=False, device=device)
    dl_valid = WavesDataloader(n_samples=100, freeze=True, device=device)
    dl_test = WavesDataloader(n_samples=100, freeze=True, freq_range=[0., 30.], device=device)  # Test generalization
    dl_dict = {
        TRAIN: DataLoader(dl_train, shuffle=True, batch_size=config[DATALOADER][BATCH_SIZE][TRAIN]),
        VALIDATION: DataLoader(dl_valid, shuffle=False, batch_size=config[DATALOADER][BATCH_SIZE][VALIDATION]),
        TEST: DataLoader(dl_test, shuffle=False, batch_size=config[DATALOADER][BATCH_SIZE][TEST])
    }
    return dl_dict


if __name__ == "__main__":
    config = {
        DATALOADER: {
            BATCH_SIZE: {
                TRAIN: 4,
                VALIDATION: 8,
                TEST: 8
            }
        }
    }
    import matplotlib.pyplot as plt
    dl_dict = get_dataloaders(config, factor=1)
    for run_index in range(2):
        for idx, mode in enumerate([TRAIN, VALIDATION, TEST]):
            signals, freqs = next(iter(dl_dict[mode]))
            plt.subplot(2, 3, run_index*3+1+idx)
            plt.plot(signals.cpu().numpy().T)
            plt.title(mode)
    plt.show()
