from tempfile import TemporaryDirectory

import numpy as np

from .base_single_agent_dataset import BaseSingleAgentDataset


class MockSingleAgentDataset(BaseSingleAgentDataset):
    def __init__(
        self,
        num_samples=100,
        sequence_length=50,
        x_bias=10,
        y_bias=10,
        input_dim=5,
        output_dim=2,
        x_len=10,
        y_len=40,
    ):
        with TemporaryDirectory() as tmp_dir:
            self.build_data_dir(
                tmp_dir,
                num_samples,
                sequence_length,
                x_bias,
                y_bias,
                input_dim,
                output_dim,
            )
            super().__init__(tmp_dir, x_len, y_len)

    def build_data_dir(
        self,
        tmp_dir,
        num_samples,
        sequence_length,
        x_bias,
        y_bias,
        input_dim,
        output_dim,
    ):
        for i in range(num_samples):
            x = np.random.randn(sequence_length, input_dim) + x_bias
            y = np.random.randn(sequence_length, output_dim) + y_bias
            data = np.concatenate([x, y], axis=1)
            np.savetxt(f"{tmp_dir}/{i}.txt", data, delimiter=",")
