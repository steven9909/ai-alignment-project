from models.mistral import MistralModel
from dataset.processor import NTruthMLieProcessor
from dataset.loader import NTruthMLieLoader
from pathlib import Path

import torch

device = torch.device("mps")

processor = NTruthMLieProcessor(
    NTruthMLieLoader(2, 1, Path("./data/facts_true_false.csv")), MistralModel(device)
)
processor.process()
