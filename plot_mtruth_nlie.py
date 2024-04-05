
###########################################
# Plot results
###########################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Data

results = pd.read_csv("./results/multi_statement_results.csv")

categories = ['Baseline PCA', 'Mass Mean', 'PCA True/False']
m1n1_id = results[results["Dataset"] == "m=1n=1"]["Accuracy"]/150
m2n1_id = results[results["Dataset"] == "m=2n=1"]["Accuracy"]/150
m3n1_id = results[results["Dataset"] == "m=3n=1"]["Accuracy"]/100

m1n1_ood = results[results["Dataset"] == "m=1n=1_cities"]["Accuracy"]/150
m2n1_ood = results[results["Dataset"] == "m=2n=1_cities"]["Accuracy"]/150
m3n1_ood = results[results["Dataset"] == "m=3n=1_cities"]["Accuracy"]/100

settings_id = (m1n1_id, m2n1_id, m3n1_id, "in-distribution")
settings_ood = (m1n1_ood, m2n1_ood, m3n1_ood, "out-of-distribution")

# Plot
for settings in [settings_id, settings_ood]:
    m1n1, m2n1, m3n1, title = settings
    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, m1n1, width, label='m=1')
    rects2 = ax.bar(x, m2n1, width, label='m=2')
    rects3 = ax.bar(x + width, m3n1, width, label='m=3')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Reader Type ({})'.format(title))
    ax.set_xticks(x)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels(categories)
    ax.legend()

    fig.tight_layout()

    plt.savefig("./results/MTruthNLie_results_{}.png".format(title))