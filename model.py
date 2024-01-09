import torch


class BaseModel(torch.nn.Module):
    """Base class for all models with additional methods"""
    # @TODO: add factory for autoregristration of model classes.
    # @TODO: add load/save methods
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ConvModel(BaseModel):
    """A simple 1D signal convolutional model
    with pooling and pointwise convolutions to allow estimating a scalar value
    """

    def __init__(self, in_channels: int, out_channels: int = 1, conv_feats: int = 8, h_dim=4, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = torch.nn.Conv1d(in_channels, conv_feats, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = torch.nn.Conv1d(conv_feats, conv_feats, kernel_size=kernel_size, padding=kernel_size//2)
        self.non_linearity = torch.nn.ReLU()
        self.pointwise1 = torch.nn.Conv1d(conv_feats, h_dim, kernel_size=1)
        self.pointwise2 = torch.nn.Conv1d(h_dim, out_channels, kernel_size=1)
        self.conv_model = torch.nn.Sequential(self.conv1, self.non_linearity, self.conv2, self.non_linearity)
        self.pool = torch.nn.AdaptiveAvgPool1d(8)
        self.final_pool = torch.nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv_model(x)
        reduced = self.pool(y)
        freq_estim = self.pointwise2(self.non_linearity(self.pointwise1(reduced)))
        return (self.final_pool(freq_estim).squeeze(-1)).squeeze(-1)


if __name__ == "__main__":
    model = ConvModel(1, conv_feats=4, h_dim=8)
    print(f"Model #parameters {model.count_parameters()}")
    n, ch, t = 4, 1, 64
    print(model(torch.rand(n, ch, t)).shape)
