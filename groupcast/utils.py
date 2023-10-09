import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback

from teamtraj.datasets import MockSingleAgentDataset


class LossTrackerCallback(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Assuming 'train_loss_epoch' is the key that stores the average loss for an epoch
        epoch_loss = trainer.callback_metrics.get("train_loss_epoch", None)
        if epoch_loss is not None:
            self.epoch_losses.append(epoch_loss.item())


def is_loss_decreasing(losses):
    for i in range(1, len(losses)):
        if losses[i] > losses[i - 1]:
            return False
    return True


def create_mock_single_agent_dataloaders(
    num_samples=100, seq_len=10, input_dim=2, output_dim=2, batch_size=64, num_workers=4
):
    trainset = MockSingleAgentDataset(
        num_samples=num_samples,
        sequence_length=seq_len,
        input_dim=input_dim,
        output_dim=output_dim,
    )
    valset = MockSingleAgentDataset(
        num_samples=num_samples,
        sequence_length=seq_len,
        input_dim=input_dim,
        output_dim=output_dim,
    )
    testset = MockSingleAgentDataset(
        num_samples=num_samples,
        sequence_length=seq_len,
        input_dim=input_dim,
        output_dim=output_dim,
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


def check_single_agent_model(model, num_samples=100, seq_len=10, input_dim=2, output_dim=2, batch_size=64, num_workers=4):

    # Create DataLoader
    train_loader, val_loader, test_loader = create_mock_single_agent_dataloaders(
        num_samples=num_samples,
        seq_len=seq_len,
        input_dim=input_dim,
        output_dim=output_dim,
        batch_size=batch_size,
        num_workers=num_workers,
    )


    # Create a callback to track the loss
    loss_tracker = LossTrackerCallback()

    # Create a dummy logger to saving logs
    logger = pl.loggers.logger.DummyLogger()

    # Using PyTorch Lightning's trainer with no logging or saving
    max_epochs = 5
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[loss_tracker],
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    # Inside your test function:
    if not is_loss_decreasing(loss_tracker.epoch_losses):
        raise AssertionError(
            f"The loss did not decrease consistently across epochs: {loss_tracker.epoch_losses}"
        )
