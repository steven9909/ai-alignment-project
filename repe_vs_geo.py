import pickle
import torch
import numpy as np
from repe import PCARepReader
from reading_vector import compare_directions
import matplotlib.pyplot as plt
def to_rep_reader(f):
    reader = PCARepReader()
    with open(f, 'rb') as f:
        data = pickle.load(f)
        vectors = data['directions']
        del vectors[-32] # Don't want the first state
        directions = {layer: vectors[layer].numpy().reshape((1,len(vectors[layer]))).astype(np.float32) for layer in vectors.keys()}
        reader.directions = directions
    signs = {layer: [1] for layer in reader.directions.keys()}
    reader.direction_signs = signs


    return reader


probes = ["MM", "LR"]

for ProbeClass in probes:
    reader = to_rep_reader(f"./data/{ProbeClass}_facts_true_false.pkl")
    reader.save(f"mistral_{ProbeClass}_facts_true_false")

# r1 = PCARepReader.from_pretrained("mistral_MM_facts_true_false")
# r2 = PCARepReader.from_pretrained("mistral_facts_true_false")

# comp = compare_directions(r2.directions, r1.directions, r2.direction_signs, r1.direction_signs)
# plt.bar(comp.keys(), comp.values())
# plt.xlabel('Layer')
# plt.ylabel('Comparison Value')
# plt.title('Comparison of Directions')
# plt.savefig("./results/comparison_repe_MM.png")