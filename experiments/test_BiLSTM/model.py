from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F

NUM_SUBGROUP = 91
NUM_COURSES = 728

class SubgroupPredictions(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super(SubgroupPredictions, self).__init__()
        self.lstm = nn.LSTM(NUM_SUBGROUP, 
                    hidden_size,
                    bidirectional=True,
                    dropout=dropout, 
                    num_layers=num_layers)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(0.5)

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        out, (h, c) = self.lstm(batch)

        return out