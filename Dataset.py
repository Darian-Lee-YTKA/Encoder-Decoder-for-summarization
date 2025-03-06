import torch
from torch import nn


class Summary_dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.vocab = {}
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.targets = []
    def build_vocab(self, df): # this assumes the df has 2 columns, src and target 
        for row 
        