import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import dataclass

@dataclass
class ArgsModel:
    d_model: int = 512
    data_len: int = 65

class FT(nn.Module):
    def __init__(self, args: ArgsModel):
        self.x_embed = nn.Embedding(args.d_model)
        self.cls_token = nn.Embedding(args.d_model)
        self.W_cat = nn.Embedding(args.d_model)
        self.W_col = nn.Embedding(args.d_model)
        self.b_cat = nn.Embedding(args.d_model)
        self.b_col = nn.Embedding(args.d_model)

    def forward(self, x):
        num_cols = self.x_embed(x[0])
        cat_cols = self.x_embed(x[1])

        T_col = 