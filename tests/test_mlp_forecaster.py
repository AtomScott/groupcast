import pytorch_lightning as pl
import torch

from teamtraj.models import MLPForecaster
from teamtraj.utils import check_single_agent_model


def test_mlp_forecaster():
    # Setting a fixed seed for reproducibility
    torch.manual_seed(42)
    pl.seed_everything(42)

    # Parameters for mock dataset and model
    input_dim = 2
    hidden_dim = 4
    output_dim = 2
    num_layers = 1
    optimizer_params = {"lr": 0.001}

    model = MLPForecaster(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        optimizer_params=optimizer_params,
        roll_out_steps=40,
    )

    check_single_agent_model(model, num_samples=100, seq_len=50, input_dim=2, output_dim=2, batch_size=64, num_workers=4)


if __name__ == "__main__":
    test_mlp_forecaster()
