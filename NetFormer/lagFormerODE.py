import torch
from torch import nn, einsum
from torch.nn import functional as F
from torchdiffeq import odeint

from NetFormer.models import Base_mouse


class LagAttentionWeighted(nn.Module):
    """LagAttention with learnable per-lag weighting via sigmoid-gated lagFuse."""

    def __init__(self, dim_E=None, dropout=0.0, activation='none', lagMax=3):
        super().__init__()
        self.activation = activation
        self.scale = dim_E ** -0.5
        self.query_linear = nn.Linear(dim_E, dim_E, bias=False)
        self.key_linear = nn.Linear(dim_E, dim_E, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.lagMax = lagMax
        self.lagFuse = nn.Parameter(torch.zeros(lagMax + 1))

    def forward(self, x):
        # x: [B, N, T, dim_e]
        x_q = x.unfold(dimension=2, size=self.lagMax + 1, step=1)        # [B, N, T-L, dim_e, L+1]
        x_q = x_q.permute(0, 4, 2, 1, 3)                                 # [B, L+1, T-L, N, dim_e]
        B, _, t, N, _ = x_q.shape
        x_k = x[:, :, self.lagMax:].unsqueeze(1).repeat(1, self.lagMax + 1, 1, 1, 1)  # [B, L+1, N, T-L, dim_e]

        queries = self.query_linear(x_q)   # [B, L+1, T-L, N, dim_e]
        keys = self.key_linear(x_k)        # [B, L+1, N, T-L, dim_e]

        logits = einsum("b l t n d, b l m t d -> b l t n m", queries, keys)  # [B, L+1, T-L, N, N]

        if self.activation == 'softmax':
            attn = logits.softmax(dim=-1)
        elif self.activation == 'sigmoid':
            attn = torch.sigmoid(logits)
        elif self.activation == 'tanh':
            attn = torch.tanh(logits)
        else:
            attn = logits

        attn = self.attn_dropout(attn) * self.scale  # [B, L+1, T-L, N, N]

        score = torch.sigmoid(self.lagFuse)  # [L+1]

        # Global adj: mean over batch and time, then lag-weighted mean
        adj_per_lag = attn.mean(dim=(0, 2)) * score[:, None, None]  # [L+1, N, N]
        adj = adj_per_lag.mean(dim=0)                                # [N, N]

        # Per-batch attention for evaluation
        attn_per_batch = (attn * score[None, :, None, None, None]).mean(dim=(1, 2))  # [B, N, N]

        return adj, attn_per_batch


class LagFormerODEMouse(Base_mouse):
    def __init__(
        self,
        model_random_seed=42,
        predict_window_size=1,
        pred_forward=3,
        dim_E=30,
        learning_rate=1e-4,
        scheduler="cycle",
        attention_activation="none",
        weight_decay=1e-5,
        constraint_loss_weight=0,
        lagMax=3,
    ):
        super().__init__()
        self.save_hyperparameters()
        torch.manual_seed(model_random_seed)
        self.predict_window_size = predict_window_size
        self.pred_forward = pred_forward

        self.conv_in = nn.Conv2d(in_channels=1, out_channels=dim_E, kernel_size=1)
        self.conv_out = nn.Conv2d(in_channels=dim_E, out_channels=1, kernel_size=1)
        self.layer_norm = nn.LayerNorm(dim_E)

        self.attentionlayer = LagAttentionWeighted(
            dim_E=dim_E,
            activation=attention_activation,
            lagMax=lagMax,
        )

        self.uProj = nn.Linear(dim_E, dim_E)
        self.uNorm = nn.LayerNorm(dim_E)

    def _embed(self, x):
        # x: [B, N, T] -> [B, N, T, dim_E]
        x_emb = self.conv_in(x.unsqueeze(1))              # [B, dim_E, N, T]
        return self.layer_norm(x_emb.permute(0, 2, 3, 1)) # [B, N, T, dim_E]

    def forward(self, x, _):
        # x: [B, N, T]
        x_emb = self._embed(x)             # [B, N, T, dim_E]
        B, N, T, D = x_emb.shape

        adj, attn_per_batch = self.attentionlayer(x_emb)
        adj = adj * (1.0 - torch.eye(N, device=adj.device))  # zero diagonal

        z0 = x_emb[:, :, -1, :]                                     # [B, N, D]
        u = self.uNorm(self.uProj(z0.mean(dim=1, keepdim=True)))    # [B, 1, D]

        def ode_func(t, z):
            drift = torch.einsum('bnd,nm->bmd', z, adj)
            return drift + u

        if self.pred_forward > 0:
            t_back = torch.linspace(0, -1, self.pred_forward + 1, device=x.device)[1:]
            ode_back = odeint(ode_func, z0, t_back, method='rk4')       # [pred_forward, B, N, D]
            ode_back = torch.flip(ode_back, dims=[0])                   # oldest -> newest

            t_fwd = torch.linspace(0, 1, self.predict_window_size + 1, device=x.device)[1:]
            ode_fwd = odeint(ode_func, z0, t_fwd, method='rk4')         # [predict_window_size, B, N, D]

            ode_out = torch.cat([ode_back, ode_fwd], dim=0)             # [total, B, N, D]
        else:
            t_fwd = torch.linspace(0, 1, self.predict_window_size + 1, device=x.device)[1:]
            ode_out = odeint(ode_func, z0, t_fwd, method='rk4')         # [predict_window_size, B, N, D]

        # Readout: [total, B, N, D] -> [B, N, total]
        ode_out = ode_out.permute(1, 3, 2, 0)               # [B, D, N, total]
        pred = self.conv_out(ode_out).squeeze(1)            # [B, N, total]

        return pred, attn_per_batch

    def training_step(self, batch, batch_idx):
        x_full, neuron_ids, cell_type_ids, state = batch
        x_full = x_full.squeeze(0)   # [B, N, T_full]

        x_in = x_full[:, :, :-self.hparams.predict_window_size]
        x_future = x_full[:, :, -self.hparams.predict_window_size:]

        if self.pred_forward > 0:
            x_past = x_in[:, :, -self.pred_forward:]
            target = torch.cat([x_past, x_future], dim=-1)
        else:
            target = x_future

        pred, _ = self(x_in, neuron_ids)
        loss = F.mse_loss(pred, target)
        self.log("TRAIN_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_full, neuron_ids, cell_type_ids, state = batch
        x_full = x_full.squeeze(0)

        x_in = x_full[:, :, :-self.hparams.predict_window_size]
        x_future = x_full[:, :, -self.hparams.predict_window_size:]

        if self.pred_forward > 0:
            x_past = x_in[:, :, -self.pred_forward:]
            target = torch.cat([x_past, x_future], dim=-1)
        else:
            target = x_future

        pred, _ = self(x_in, neuron_ids)
        loss = F.mse_loss(pred, target)
        self.log("VAL_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x_full, neuron_ids, cell_type_ids, state = batch
        x_full = x_full.squeeze(0)
        state = state.squeeze(0)

        x_in = x_full[:, :, :-self.hparams.predict_window_size]
        x_future = x_full[:, :, -self.hparams.predict_window_size:]

        pred, attn = self(x_in, neuron_ids)                    # pred: [B, N, total]
        pred_future = pred[:, :, self.pred_forward:]           # [B, N, predict_window_size]

        return pred_future, x_future, attn, state
