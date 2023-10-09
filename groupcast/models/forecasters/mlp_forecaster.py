from torch import nn

from ..base_forecaster import BaseForecaster


class MLPForecaster(BaseForecaster):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        optimizer_params={"lr": 0.001},
        scheduler_params={"T_0": 10, "eta_min": 1e-6},
        loss_fn=nn.MSELoss(),
        roll_out_steps=10,
        original_fps=25,
        inference_fps=25,
    ):
        super().__init__(
            optimizer_params=optimizer_params,
            scheduler_params=scheduler_params,
            loss_fn=loss_fn,
            roll_out_steps=roll_out_steps,
            original_fps=original_fps,
            inference_fps=inference_fps,
        )

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim  # Set input dim for next layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)
