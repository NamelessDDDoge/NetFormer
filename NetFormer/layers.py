import torch
from torch import nn, einsum
from torch.nn import functional as F


class Attention(nn.Module):
    """
    Attention layer in NetFormer, which takes input tensor x and entity tensor e, and returns the output tensor.
    """

    def __init__(
        self,
        dim_X,
        dim_E,
        *,
        dropout=0.0,
        activation='none', # 'sigmoid' or 'tanh' or 'softmax' or 'none'
    ):
        super().__init__()
        self.activation = activation

        self.scale = (dim_X + dim_E) ** -0.5

        # Q, K

        self.query_linear = nn.Linear(dim_X + dim_E, dim_X + dim_E, bias=False)
        self.key_linear = nn.Linear(dim_X + dim_E, dim_X + dim_E, bias=False)

        # dropouts

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, e):

        x_e = torch.cat((x, e), dim=-1)

        batch_size, n, t = x.shape

        # We_Q_We_KT: (dim_E, dim_E)

        # We_Q_We_KT = (self.query_linear.weight.clone().detach().T)[t:] @ (self.key_linear.weight.clone().detach().T)[t:].T
        # attn3 = einsum("b n e, b m e -> b n m", e @ We_Q_We_KT, e)

        # Q, K

        queries = self.query_linear(x_e)
        keys = self.key_linear(x_e)

        logits = einsum("b n d, b m d -> b n m", queries, keys)
        if self.activation == 'softmax':
            attn = logits.softmax(dim=-1)
        elif self.activation == 'sigmoid':
            attn = F.sigmoid(logits)
        elif self.activation == 'tanh':
            attn = F.tanh(logits)
        elif self.activation == 'none':
            attn = logits

        attn = self.attn_dropout(attn)
        attn = attn * self.scale

        v = x  # identity mapping
        out = einsum("b n m, b m t -> b n t", attn, v)

        out = out + x   # residual connection
        return out, attn
    