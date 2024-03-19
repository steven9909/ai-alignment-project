from models.mistral import MistralModel
from dataset.processor import NTruthMLieProcessor
from dataset.loader import NTruthMLieLoader
from pathlib import Path

import torch

processor = NTruthMLieProcessor(
    NTruthMLieLoader(2, 2, Path("./data/facts_true_false.csv")),
    MistralModel(torch.device("mps")),
    torch.device("mps"),
)
processor.process()
