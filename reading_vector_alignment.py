import matplotlib.pyplot as plt
import numpy as np
from reading_vector import compare_directions
from repe import PCARepReader

readers = ["mistral_facts_true_false", 
           "mistral_nonsense",
           "mistral_m0n1",
            "mistral_MM_facts_true_false",
            "mistral_LR_facts_true_false",
            "mistral_facts_true_false_statement_contrast", 
            "mistral_facts_true_false_statement-prompt_contrast", 
            ]

ax_labels = ["baseline", "random", "inverted", "MM", "LR", "TF", "prompt-TF", ]
layers = range(-10, -25, -1)

avg_alignments = np.eye(len(readers))
# mid_layer_alignments = np.eye(len(readers))
for i in range(len(readers)):
    for j in range(i+1, len(readers)):
        reader1 = PCARepReader.from_pretrained(readers[i])
        reader2 = PCARepReader.from_pretrained(readers[j])

        v1, s1 = reader1.directions, reader1.direction_signs
        v2, s2 = reader2.directions, reader2.direction_signs
        comparisons = compare_directions(v1, v2, s1, s2)
        avg = np.mean([comparisons[layer] for layer in layers])
        avg_alignments[i, j] = avg
        avg_alignments[j, i] = avg
        # mid_layer_alignments[i, j] = comparisons[-20]
        # mid_layer_alignments[j, i] = comparisons[-20]

# for array in [avg_alignments, mid_layer_alignments]:

fig, ax = plt.subplots(figsize=(10,10))
# fig, ax = plt.subplots(1, 2, figsize=(12, 24))
heatmap1 = ax.imshow(avg_alignments, cmap='hot', interpolation='nearest')
# heatmap2 = ax[1].imshow(mid_layer_alignments, cmap='hot', interpolation='nearest')

plt.colorbar(heatmap1, ax=ax)
ax.set_title('Average Alignments - Layers {}-{}'.format(31+layers[-1], 31+layers[0]))
ax.set_xticks(range(len(readers)), ax_labels, rotation=90)
ax.set_yticks(range(len(readers)), ax_labels)
# for a in ax:
#     a.set_xticks(range(len(readers)), ax_labels, rotation=90)
#     a.set_yticks(range(len(readers)), ax_labels)
# ax[0].set_title('Average Alignments')
# ax[1].set_title('Layer 20 Alignments')
fig.savefig("./results/average_alignments.png")
