import numpy as np
import torch

from .base_dataset import BaseDataset


class BaseMultiAgentDataset(BaseDataset):
    def __init__(self, data_dir, x_len, y_len, x_transform=None, y_transform=None):
        super().__init__()
        self.data = []
        self.data_dir = data_dir
        self.x_len = x_len
        self.y_len = y_len
        self.x_transform = x_transform
        self.y_transform = y_transform

        # load data to self.data
        # self.data is a list of ararys shape (seq_len, 2)
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        if self.x_transform:
            data = self.x_transform(data)

        if self.y_transform:
            data = self.y_transform(data)

        return data

    def load_data(self):
        for file in self.files:
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

    def get_files(self, data_dir):
        files = []
        for file in data_dir.glob("*.txt"):
            files.append(file)
        return files

    def collate_fn(self, batch, max_num_agents, dummy_value=-1000):
        len(batch)
        x, y = [], []
        for i, (x_seq, y_seq) in enumerate(batch):
            x_shape = (x_seq.shape[0], max_num_agents, x_seq.shape[-1])
            y_shape = (y_seq.shape[0], max_num_agents, y_seq.shape[-1])

            x_full = torch.full(
                x_shape, dummy_value, dtype=x_seq.dtype, device=x_seq.device
            )
            y_full = torch.full(
                y_shape, dummy_value, dtype=y_seq.dtype, device=y_seq.device
            )

            num_agents_x = min(x_seq.shape[1], max_num_agents)
            num_agents_y = min(y_seq.shape[1], max_num_agents)

            x_full[:, :num_agents_x, :] = x_seq[:, :num_agents_x, :]
            y_full[:, :num_agents_y, :] = y_seq[:, :num_agents_y, :]

            x.append(x_full)
            y.append(y_full)
        return torch.stack(x), torch.stack(y)
