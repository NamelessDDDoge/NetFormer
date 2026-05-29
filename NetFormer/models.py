import numpy as np
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from NetFormer.layers import Attention



####################################################################################################
# NetFormer model for connectivity-constrained simulation data
####################################################################################################

class Base_sim(pl.LightningModule):
    def __init__(self,) -> None:
        super().__init__()
        self.save_hyperparameters()

    def forward(self, x):
        return NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

        if self.hparams.scheduler == "plateau":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=6,
                ),
                "monitor": "VAL_loss",
            }
        elif self.hparams.scheduler == "cycle":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=self.hparams.learning_rate / 2,
                    max_lr=self.hparams.learning_rate * 2,
                    cycle_momentum=False,
                ),
                "interval": "step",
            }
        else:
            print("No scheduler is used")
            return [optimizer]

        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, neuron_level_attention = self(x)
        
        pred = y_hat
        target = y

        loss = F.mse_loss(pred, target, reduction="mean")
        self.log("TRAIN_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, neuron_level_attention = self(x)
        
        pred = y_hat
        target = y

        loss = F.mse_loss(pred, target, reduction="mean")
        self.log("VAL_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat, neuron_level_attention = self(x)

        return y_hat, y, neuron_level_attention
    

class NetFormer_sim(Base_sim):
    def __init__(
        self,
        model_random_seed=42,
        neuron_num=200,
        window_size=200,
        learning_rate=1e-4,
        scheduler="plateau",
        predict_window_size=1,
        attention_activation = "none", # "softmax" or "sigmoid" or "tanh", "none"
        weight_decay = 0,
        out_layer=False,
        dim_E=100,
    ):
        super().__init__()
        self.save_hyperparameters()
        torch.manual_seed(model_random_seed)

        self.predict_window_size = predict_window_size

        dim_X = window_size - predict_window_size

        # Embedding

        self.embedding_table = nn.Embedding(
            num_embeddings=neuron_num, embedding_dim=dim_E
        )

        self.layer_norm = nn.LayerNorm(dim_X+dim_E)

        # Attention

        self.attentionlayer = Attention(
            dim_X=dim_X,
            dim_E=dim_E,
            activation=attention_activation,
        )

        self.layer_norm2 = nn.LayerNorm(dim_X)

        self.out_layer = out_layer
        if out_layer == True:
            self.out = nn.Linear(dim_X, predict_window_size, bias=False)

    def forward(self, x):

        idx = torch.arange(x.shape[1]).to(x.device)
        e = self.embedding_table(idx)
        e = e.repeat(x.shape[0],1,1)  # (m, e) to (b, m, e)

        x_e = self.layer_norm(torch.cat((x, e), dim=-1))
        # Split x and e
        x = x_e[:, :, :x.shape[-1]]
        e = x_e[:, :, x.shape[-1]:]

        x, attn = self.attentionlayer(x, e)
        x = self.layer_norm2(x)

        if self.out_layer:
            return self.out(x), attn
        else:
            return x[:, :, -1*self.predict_window_size:], attn
        




####################################################################################################
# NetFormer model for multimodal mouse data
####################################################################################################

class Base_mouse(pl.LightningModule):
    def __init__(self,) -> None:
        super().__init__()
        self.save_hyperparameters()

    def forward(self, x):
        return NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

        if self.hparams.scheduler == "plateau":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=3,
                ),
                "monitor": "VAL_loss",
            }
        elif self.hparams.scheduler == "cycle":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=self.hparams.learning_rate / 2,
                    max_lr=self.hparams.learning_rate * 2,
                    cycle_momentum=False,
                ),
                "interval": "step",
            }
        else:
            print("No scheduler is used")

        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        x, neuron_ids, cell_type_ids, state = batch
        x = x.squeeze(0)                 # remove the fake batch_size
        neuron_ids = neuron_ids.squeeze(0)
        cell_type_ids = cell_type_ids.squeeze(0)

        # make the last time step as the target
        target = x[:, :, -1*self.hparams.predict_window_size:].clone()
        pred, neuron_level_attention = self(x[:, :, :-1*self.hparams.predict_window_size], neuron_ids)

        # make pred and target have the same shape
        target = target.reshape(pred.shape)

        if self.hparams.constraint_loss_weight != 0:
            # cell type level constraint
            cell_type_ids_np = cell_type_ids[0].clone().detach().cpu().numpy()
            expanded_cell_type_level_constraint = torch.zeros((neuron_level_attention.shape[1],neuron_level_attention.shape[2]), requires_grad=True).to(pred.device)
            expanded_cell_type_level_var = torch.zeros((neuron_level_attention.shape[1],neuron_level_attention.shape[2]), requires_grad=True).to(pred.device)

            # loop over unique cell types
            for i in list(np.unique(cell_type_ids_np)):
                # find the neurons with the same cell type
                neuron_ids_with_same_cell_type_i = np.where(cell_type_ids_np == i)[0]
                for j in list(np.unique(cell_type_ids_np)):
                    # find the neurons with the same cell type
                    neuron_ids_with_same_cell_type_j = np.where(cell_type_ids_np == j)[0]
                    # Assign the same constraint to the neurons with the same cell type
                    expanded_cell_type_level_constraint[neuron_ids_with_same_cell_type_i, neuron_ids_with_same_cell_type_j.reshape(-1,1)] = self.cell_type_level_constraint[i, j] * 1   # to create computational graph
                    expanded_cell_type_level_constraint[neuron_ids_with_same_cell_type_j, neuron_ids_with_same_cell_type_i.reshape(-1,1)] = self.cell_type_level_constraint[j, i] * 1

                    expanded_cell_type_level_var[neuron_ids_with_same_cell_type_i, neuron_ids_with_same_cell_type_j.reshape(-1,1)] = self.cell_type_level_var[i, j] ** 2   # to create computational graph
                    expanded_cell_type_level_var[neuron_ids_with_same_cell_type_j, neuron_ids_with_same_cell_type_i.reshape(-1,1)] = self.cell_type_level_var[j, i] ** 2

            # expand the first dimension of expanded_cell_type_level_constraint to batch_size
            expanded_cell_type_level_constraint = einops.repeat(expanded_cell_type_level_constraint, 'n d -> b n d', b=neuron_level_attention.shape[0])
            expanded_cell_type_level_var = einops.repeat(expanded_cell_type_level_var, 'n d -> b n d', b=neuron_level_attention.shape[0])

            # loss
            constraint_loss = F.gaussian_nll_loss(neuron_level_attention, expanded_cell_type_level_constraint, reduction="mean", var=expanded_cell_type_level_var)
            loss = F.mse_loss(pred, target, reduction="mean") + constraint_loss * self.hparams.constraint_loss_weight
        else:
            loss = F.mse_loss(pred, target, reduction="mean")

        self.log("TRAIN_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, neuron_ids, cell_type_ids, state = batch
        x = x.squeeze(0)                 # remove the fake batch_size
        neuron_ids = neuron_ids.squeeze(0)
        cell_type_ids = cell_type_ids.squeeze(0)

        # Make the last time step as the target
        target = x[:, :, -1*self.hparams.predict_window_size:].clone()
        pred, neuron_level_attention = self(x[:, :, :-1*self.hparams.predict_window_size], neuron_ids)

        # Make pred and target have the same shape
        target = target.reshape(pred.shape)

        if self.hparams.constraint_loss_weight != 0:
            # cell type level constraint
            cell_type_ids_np = cell_type_ids[0].clone().detach().cpu().numpy()
            expanded_cell_type_level_constraint = torch.zeros((neuron_level_attention.shape[1],neuron_level_attention.shape[2]), requires_grad=True).to(pred.device)
            expanded_cell_type_level_var = torch.zeros((neuron_level_attention.shape[1],neuron_level_attention.shape[2]), requires_grad=True).to(pred.device)

            # loop over unique cell types
            for i in list(np.unique(cell_type_ids_np)):
                # find the neurons with the same cell type
                neuron_ids_with_same_cell_type_i = np.where(cell_type_ids_np == i)[0]
                for j in list(np.unique(cell_type_ids_np)):
                    # find the neurons with the same cell type
                    neuron_ids_with_same_cell_type_j = np.where(cell_type_ids_np == j)[0]
                    # Assign the same constraint to the neurons with the same cell type
                    expanded_cell_type_level_constraint[neuron_ids_with_same_cell_type_i, neuron_ids_with_same_cell_type_j.reshape(-1,1)] = self.cell_type_level_constraint[i, j] * 1
                    expanded_cell_type_level_constraint[neuron_ids_with_same_cell_type_j, neuron_ids_with_same_cell_type_i.reshape(-1,1)] = self.cell_type_level_constraint[j, i] * 1

                    expanded_cell_type_level_var[neuron_ids_with_same_cell_type_i, neuron_ids_with_same_cell_type_j.reshape(-1,1)] = self.cell_type_level_var[i, j] ** 2
                    expanded_cell_type_level_var[neuron_ids_with_same_cell_type_j, neuron_ids_with_same_cell_type_i.reshape(-1,1)] = self.cell_type_level_var[j, i] ** 2

            # expand the first dimension of expanded_cell_type_level_constraint to batch_size
            expanded_cell_type_level_constraint = einops.repeat(expanded_cell_type_level_constraint, 'n d -> b n d', b=neuron_level_attention.shape[0])
            expanded_cell_type_level_var = einops.repeat(expanded_cell_type_level_var, 'n d -> b n d', b=neuron_level_attention.shape[0])

            # loss
            constraint_loss = F.gaussian_nll_loss(neuron_level_attention, expanded_cell_type_level_constraint, reduction="mean", var=expanded_cell_type_level_var)
            loss = F.mse_loss(pred, target, reduction="mean") + constraint_loss * self.hparams.constraint_loss_weight
        else:
            loss = F.mse_loss(pred, target, reduction="mean")

        self.log("VAL_loss", loss)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, neuron_ids, cell_type_ids, state = batch
        x = x.squeeze(0)                 # remove the fake batch_size
        neuron_ids = neuron_ids.squeeze(0)
        cell_type_ids = cell_type_ids.squeeze(0)
        state = state.squeeze(0)

        # Make the last time step as the target
        target = x[:, :, -1*self.hparams.predict_window_size:].clone()
        pred, neuron_level_attention = self(x[:, :, :-1*self.hparams.predict_window_size], neuron_ids)
        return pred, target, neuron_level_attention, state
    

class NetFormer_mouse(Base_mouse):
    def __init__(
        self,
        num_unqiue_neurons,
        num_cell_types,
        model_random_seed=42,
        window_size=200,
        predict_window_size = 1,
        learning_rate=1e-4,
        scheduler="cycle",
        attention_activation="none", # "softmax" or "sigmoid" or "tanh", "none"
        weight_decay=0,
        dim_E=8,
        constraint_loss_weight=0,
    ):
        super().__init__()
        self.save_hyperparameters()
        torch.manual_seed(model_random_seed)

        self.cell_type_level_constraint = nn.Parameter(torch.FloatTensor(num_cell_types, num_cell_types).uniform_(-1, 1))
        self.cell_type_level_var = nn.Parameter(torch.ones(num_cell_types, num_cell_types), requires_grad=True)

        self.predict_window_size = predict_window_size
        dim_X = window_size - predict_window_size

        # Embedding

        self.embedding_table = nn.Embedding(
            num_embeddings=num_unqiue_neurons, embedding_dim=dim_E   # global unique neuron lookup table
        )

        self.layer_norm = nn.LayerNorm(dim_X+dim_E)

        # Attention

        self.attentionlayer = Attention(
            dim_X=dim_X,
            dim_E=dim_E,
            activation=attention_activation,
        )

        self.layer_norm2 = nn.LayerNorm(dim_X)

    def forward(self, x, neuron_ids): # x: (batch_size, neuron_num, time), neuron_ids: (batch_size, neuron_num)

        e = self.embedding_table(neuron_ids[0])
        e = e.repeat(x.shape[0],1,1)  # (m, e) to (b, m, e)

        x_e = self.layer_norm(torch.cat((x, e), dim=-1))
        # Split x and e
        x = x_e[:, :, :x.shape[-1]]
        e = x_e[:, :, x.shape[-1]:]

        x, attn = self.attentionlayer(x, e)
        x = self.layer_norm2(x)

        return x[:, :, -1*self.predict_window_size:], attn