import warnings
from collections import Counter
from statistics import mean

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping

import lightning.pytorch as pl
from lightning.pytorch import Trainer
import wandb
from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn
import torch.nn.functional as F

from sampler import ImbalancedDatasetSampler
from sklearn.metrics import balanced_accuracy_score, mean_squared_error


def get_mse_macro(y_true, y_pred):
    mse_each_class = []
    for true_class in set(y_true):
        tt, pp = zip(*[[tt, pp] for tt, pp in zip(y_true, y_pred) if tt == true_class])
        mse_each_class.append(mean_squared_error(y_true=tt, y_pred=pp))
    return mean(mse_each_class)


class TraditionalLinear(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(TraditionalLinear, self).__init__()
        self.dropout = dropout
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.sigmoid(self.fc(x))


class Rubricnet(nn.Module):
    def __init__(self, descriptor_input_sizes, num_classes, dropout):
        super(Rubricnet, self).__init__()
        # Initialize linear layers for each descriptor. Assuming each input descriptor is a single value.
        self.descriptor_layers = nn.ModuleList([nn.Linear(1, 1) for _ in range(descriptor_input_sizes)])
        # Final layer to map aggregated score to ordinal classes. Adjusts to take a single aggregated score.
        self.final_layer = nn.Linear(1, num_classes)
        self.dropout = dropout

    def forward(self, descriptors):
        if self.training:
            descriptors = F.dropout(descriptors, p=self.dropout)
        # Process each descriptor through its layer, apply tanh, and scale output
        # scores = [torch.sigmoid(layer(descriptors[:, idx].unsqueeze(-1))) for idx, layer in enumerate(self.descriptor_layers)]
        scores = [torch.tanh(layer(descriptors[:, idx].unsqueeze(-1))) for idx, layer in
                  enumerate(self.descriptor_layers)]
        # Sum the scores to get a single aggregated score tensor
        aggregated_score = torch.sum(torch.stack(scores), dim=0)  # Ensure correct summation over batch
        # Pass the aggregated score through the final layer to get logits for the classes
        logits = self.final_layer(aggregated_score)  # Correct shape for linear layer input
        # Apply sigmoid to map logits to probabilities
        probabilities = torch.sigmoid(logits)
        #probabilities = F.log_softmax(logits, dim=1)
        return probabilities

    def get_regression_values(self, descriptors):
        if self.training:
            descriptors = F.dropout(descriptors, p=self.dropout)
        # Process each descriptor through its layer, apply tanh, and scale output
        scores = [torch.tanh(layer(descriptors[:, idx].unsqueeze(-1))) for idx, layer in
                  enumerate(self.descriptor_layers)]
        # Sum the scores to get a single aggregated score tensor
        aggregated_score = torch.sum(torch.stack(scores), dim=0)
        return aggregated_score

    def get_descriptor_scores(self, descriptors):
        if self.training:
            descriptors = F.dropout(descriptors, p=self.dropout)
        # Process each descriptor through its layer, apply tanh, and scale output
        scores = [torch.tanh(layer(descriptors[:, idx].unsqueeze(-1))).detach().cpu().squeeze() for idx, layer in
                  enumerate(self.descriptor_layers)]
        return scores


def _prediction2label(pred):
    """Convert ordinal predictions to class labels."""
    return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1
    #return torch.argmax(pred, dim=1)


class OrdinalLoss(nn.Module):
    """Ordinal regression with encoding as in https://arxiv.org/pdf/0704.1028.pdf"""

    def __init__(self, weight_class=None):
        super(OrdinalLoss, self).__init__()
        self.weights = weight_class

    def forward(self, predictions, targets):
        # Fill in ordinal target function, i.e., 0 -> [1,0,0,...]
        modified_target = torch.zeros_like(predictions)
        for i, target in enumerate(targets):
            modified_target[i, 0:target + 1] = 1

        # if torch tensor is empty, return 0
        if predictions.shape[0] == 0:
            return 0
        # loss
        if self.weights is not None:
            self.weights = self.weights.to(predictions.device)
            return torch.sum((self.weights * F.mse_loss(predictions, modified_target, reduction="none")).mean(axis=1))
        else:
            return torch.sum(F.mse_loss(predictions, modified_target, reduction="none").mean(axis=1))



class LogisticRegressionOrdinal(pl.LightningModule):
    def __init__(self, input_dim, num_classes, lr, hidden_size, num_layers, dropout, decay_lr, weight_decay):
        super(LogisticRegressionOrdinal, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lr = lr
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.decay_lr = decay_lr
        self.weight_decay = weight_decay
        # self.linear = torch.nn.Linear(input_dim, num_classes)
        self.linear1 = Rubricnet(input_dim, num_classes, dropout)
        self.loss_fn = OrdinalLoss()
        #self.loss_fn = torch.nn.NLLLoss()
        self.learning_rate = lr

    def forward(self, x):
        return self.linear1(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y.long())
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_logits = self(x)
        y_pred = _prediction2label(y_logits)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            acc = balanced_accuracy_score(y.cpu().detach(), y_pred.cpu().detach())
            mse = get_mse_macro(y.cpu().detach(), y_pred.cpu().detach())
            loss = self.loss_fn(y_logits, y.long())

        # Identify the dataset being evaluated
        if dataloader_idx == 0:  # Training dataset evaluation
            metric_prefix = 'train'
        elif dataloader_idx == 1:  # Validation dataset
            metric_prefix = 'val'
        elif dataloader_idx == 2:  # Testing dataset
            metric_prefix = 'test'
        else:
            raise ValueError(f"Unexpected dataloader_idx: {dataloader_idx}")

        # Log metrics
        self.log(f'loss/{metric_prefix}', loss, logger=True)
        self.log(f'acc/{metric_prefix}', acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'MSE/{metric_prefix}', mse, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=self.decay_lr)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-1,
        #                      step_size_up=5, step_size_down=20,
        #                      mode='triangular', cycle_momentum=False)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'acc/val/dataloader_idx_1',  # Specify the metric to monitor
                'interval': 'epoch',
                'frequency': 1,
                'strict': True,
            }
        }

    def on_train_epoch_start(self):
        # Log the current learning rate
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', lr, on_epoch=True, prog_bar=False, logger=True)


class RubricnetSklearn:
    def __init__(self, input_dim, num_classes, split=0, args=None, logging=True):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.decay_lr = args.decay_lr
        self.weight_decay = args.weight_decay
        model = LogisticRegressionOrdinal(input_dim, num_classes, lr=args.lr, hidden_size=args.hidden_size,
                                          num_layers=args.num_layers, dropout=args.dropout, decay_lr=args.decay_lr,
                                          weight_decay=args.weight_decay)
        self.model = model
        early_stopping = EarlyStopping(
            monitor='acc/val/dataloader_idx_1',
            patience=args.patience,
            verbose=True,
            mode='max'
        )
        if logging:
            wandb_logger = WandbLogger(
                # set the wandb project where this run will be logged
                project="explainable",
                name="split_number_" + str(split),
                # track hyperparameters and run metadata
                entity="anonymous",
                group=args.alias_experiment,
                log_model="all"
            )
            wandb_logger.log_hyperparams(vars(args))
        # Initialize the ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor='acc/val/dataloader_idx_1',  # Metric to monitor
            dirpath=f'checkpoints/{args.alias_experiment}',  # Directory where checkpoints will be saved
            filename=f'split_{split}',  # File name for the checkpoint
            save_top_k=1,  # Save only the best checkpoint
            mode='max',  # The `min` mode will save the checkpoint with the minimum `val_loss`
        )
        self.trainer = Trainer(
            max_epochs=10000,
            logger=wandb_logger if logging else None,
            callbacks=[early_stopping, checkpoint_callback],
            enable_progress_bar=True
        )

    def load_model(self, path):
        self.model = LogisticRegressionOrdinal.load_from_checkpoint(
            path, input_dim=self.input_dim, num_classes=self.num_classes, lr=self.lr,
            hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout,
            decay_lr=self.decay_lr, weight_decay=self.weight_decay
        ).to(self.model.device)

    def calculate_weights(self, y_train, num_classes=9):
        count = dict(Counter(y_train.tolist()))
        weight_class = []
        for ii in range(num_classes):
            weight_class.append(count.get(ii, 0))  # Use .get to handle missing classes gracefully
        weight_class = torch.tensor(weight_class, dtype=torch.float32)
        weight_class = weight_class / weight_class.sum()  # Normalize
        weight_class = 1.0 / (weight_class + 1e-6)  # Avoid division by zero and normalize
        self.model.loss_fn.weights = weight_class  # Assign weights to the loss function
        self.model.loss_fn.weights = self.model.loss_fn.weights.to(
            self.model.device)  # Move to the same device as the model

    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
        self.calculate_weights(y_train_tensor)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32)

        # Now use TensorDataset with the tensors
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        def get_label_callback(dataset):
            # Extracts labels for the ImbalancedDatasetSampler
            labels = [label.item() for _, label in dataset]
            return torch.tensor(labels)

        # Prepare DataLoaders
        # Initialize the sampler
        sampler = ImbalancedDatasetSampler(
            dataset=train_dataset,
            callback_get_label=get_label_callback,
            modify_weights=True  # Or False, depending on your needs
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler)
        train_loader_entire = DataLoader(train_dataset, batch_size=len(train_dataset))
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        # Train the model with validation support
        self.trainer.fit(self.model, train_dataloaders=train_loader,
                         val_dataloaders=[train_loader_entire, val_loader, test_loader])
        wandb.finish()

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return _prediction2label(predictions)
        #return torch.argmax(predictions, dim=1)

    def predict_regression(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32)
        regression = self.model.linear1.get_regression_values(X)
        return regression.detach().cpu().squeeze()

    def predict_descriptor_scores(self, X_test_scaled):
        self.model.eval()
        X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        with torch.no_grad():
            scores = self.model.linear1.get_descriptor_scores(X_tensor)
        return scores
