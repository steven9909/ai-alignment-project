from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
import csv

from repe import repe_pipeline_registry, PCARepReader

repe_pipeline_registry()

from honesty.utils import (
    honesty_function_dataset,
    plot_lat_scans,
    plot_detection_results,
)

model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
dataset_name = "facts_true_false"
device = (
    "cuda"
    if torch.cuda.is_available()
    else (torch.device("mps") if torch.backends.mps.is_available() else "cpu")
)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.float16, device_map=device
)
use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False
)
tokenizer.pad_token_id = 0

rep_token = -1
hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
n_difference = 1
direction_method = "pca"
rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)

user_tag = "[INST]"
assistant_tag = "[/INST]"

data_path = f"./data/{dataset_name}.csv"
dataset = honesty_function_dataset(data_path, tokenizer, user_tag, assistant_tag)

honesty_rep_reader = rep_reading_pipeline.get_directions(
    dataset["train"]["data"],
    rep_token=rep_token,
    hidden_layers=hidden_layers,
    n_difference=n_difference,
    train_labels=dataset["train"]["labels"],
    direction_method=direction_method,
    batch_size=32,
)

data = csv.reader(open("./data/activations/association.csv", "r"), delimiter=",")


for i, row in enumerate(data):
    # Process each row of data here
    sentence = row[1]
    labels = row[2]

    H_test = rep_reading_pipeline(
        [sentence],
        hidden_layers=hidden_layers,
        rep_token=-1,
        rep_reader=honesty_rep_reader,
    )

    print(H_test)
