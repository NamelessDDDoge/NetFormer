import torch
from torch import nn, einsum
from torch.nn import functional as F

from NetFormer.models import Base_mouse

class LagAttention(nn.Module):
    """
    Lag Attention layer
    """

    def __init__(
        self,
        dim_X=None,
        dim_E=None,
        dropout=0.0,
        activation='none', # 'sigmoid' or 'tanh' or 'softmax' or 'none',
        lagMax=3
    ):
        super().__init__()
        self.activation = activation

        self.scale = dim_E ** -0.5

        # Q, K

        self.query_linear = nn.Linear(dim_E, dim_E, bias=False)
        self.key_linear = nn.Linear(dim_E, dim_E, bias=False)

        # dropouts

        self.attn_dropout = nn.Dropout(dropout)
        self.lagMax = lagMax

    def forward(self, x):
        # x: [B, N, T, dim_e]
        x_q = x.unfold(dimension=2, size=self.lagMax+1, step=1)      
        x_q = x_q.permute(0, 4, 2, 1, 3)                                                # [B, L, T-L, N, dim_e]

        B, _, t, N, _ = x_q.shape
        x_k = x[:, :, self.lagMax:].unsqueeze(1).repeat(1, self.lagMax+1, 1, 1, 1)      # [B, L, N, T-L, dim_e]

        queries = self.query_linear(x_q)            # [B, L, T-L, N, dim_e]
        keys = self.key_linear(x_k)                 # [B, L, N, T-L, dim_e]

        logits = einsum("b l t n d, b l m t d -> b l t n m", queries, keys)

        if self.activation == 'softmax':
            attn = logits.softmax(dim=-1)
        elif self.activation == 'sigmoid':
            attn = F.sigmoid(logits)
        elif self.activation == 'tanh':
            attn = F.tanh(logits)
        elif self.activation == 'none':
            attn = logits

        attn = self.attn_dropout(attn)          # [B, L, T-L, N, N]
        attn = attn * self.scale
        attn = attn.mean(dim=(1, 2))

        v = x  # identity mapping
        out = einsum("b n m, b m t d -> b n t d", attn, v)

        out = out + x   # residual connection
        return out, attn
    

class LagFormerMouse(Base_mouse):
    def __init__(self, 
                 model_random_seed=42, 
                 predict_window_size=1,
                 dim_E=30,
                 learning_rate=1e-4,
                 scheduler="cycle",
                 attention_activation="none",
                 weight_decay=1e-5,
                 constraint_loss_weight=0,
                 ):
        super().__init__()
        self.save_hyperparameters()
        torch.manual_seed(model_random_seed)
        self.predict_window_size = predict_window_size

        self.conv_in = nn.Conv2d(in_channels=1, out_channels=dim_E, kernel_size=1)
        self.conv_out = nn.Conv2d(in_channels=dim_E, out_channels=1, kernel_size=1) 
        self.layer_norm = nn.LayerNorm(dim_E)

        self.attentionlayer = LagAttention(
            dim_E=dim_E, 
            activation=attention_activation, 
            lagMax=3
        )

        self.layer_norm2 = nn.LayerNorm(dim_E)

    def forward(self, x, _):
        # x: [B, N, T]
        x_emb = self.conv_in(x.unsqueeze(1))                    # [B, dim_E, N, T]
        x_emb = self.layer_norm(x_emb.permute(0, 2, 3, 1))      # [B, N, T, dim_E]
        
        x, attn = self.attentionlayer(x_emb)
        x = self.layer_norm2(x)                                 # [B, N, T, dim_E]
        x = self.conv_out(x.permute(0, 3, 1, 2)).squeeze(1)     # [B, N, T]

        return x[:, :, -1*self.predict_window_size:], attn
