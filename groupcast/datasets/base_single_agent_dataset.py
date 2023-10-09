from pathlib import Path

import numpy as np
import torch
from .base_dataset import BaseDataset
from tqdm import tqdm


class BaseSingleAgentDataset(BaseDataset):
    def __init__(self, data_dir, x_len, y_len, x_transform=None, y_transform=None):
        super().__init__()
        self.data = []
        self.data_dir = Path(data_dir)
        self.x_len = x_len
        self.y_len = y_len
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.files = self.get_files()
        assert len(self.files) > 0, f"Found no files in {data_dir}"

        # load data to self.data
        # self.data is a list of ararys shape (seq_len, 2)
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = torch.tensor(data, dtype=torch.float32)

        x_data = data[: self.x_len]
        y_data = data[self.x_len : self.x_len + self.y_len]

        if self.x_transform:
            x_data = self.x_transform(x_data)

        if self.y_transform:
            y_data = self.y_transform(y_data)

        return x_data, y_data

    def load_data(self):
        for file in tqdm(self.files, desc="Loading data"):
            file_data = np.loadtxt(file, delimiter=",")

            # Reshape data to (seq_len, num_players, 2)
            n_players = file_data.shape[1] // 2
            x_player_indices = np.arange(0, n_players * 2, 2)
            y_player_indices = np.arange(1, n_players * 2, 2)
            file_data = np.stack(
                [file_data[:, x_player_indices], file_data[:, y_player_indices]],
                axis=-1,
            )

            for player_i in range(n_players):
                self.data.append(file_data[:, player_i, :])

    def get_files(self):
        files = []
        for file in self.data_dir.glob("*.txt"):
            files.append(file)
        return files
