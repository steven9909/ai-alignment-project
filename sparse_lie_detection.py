from reading_vector import NTruthMLieExperiment
import pandas as pd
import numpy as np

def CE_loss(prediction, label):
    return -(label*np.log(prediction) + (1-label)*np.log(1-prediction))


model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
experiment = NTruthMLieExperiment(model_name_or_path)

readers = ["mistral_facts_true_false", 
            "mistral_MM_facts_true_false",
            "mistral_facts_true_false_statement_contrast", 
            ]
datasets = ["m=1n=1",
            "m=1n=1_cities",
            "m=2n=1",
            "m=2n=1_cities", 
            "m=3n=1",
            "m=3n=1_cities"]

combinations = [(reader, dataset) for reader in readers for dataset in datasets]

results = pd.DataFrame(columns=['Reader', 'Dataset', 'Accuracy'])

for reader, dataset in combinations:
    print("###########################\n", reader, dataset)
    reader_type = "cluster" if "MM" in reader else "pca"
    experiment.load_rep_reader(reader, reader_type)


    data = pd.read_csv(f"./data/{dataset}.csv", dtype={'truth':str}).sample(3)
    sentences = data["statement"]
    truth_vals = data["truth"]
    labels = [[float(label) for label in val] for val in truth_vals]
    test_data = list(sentences)

    layers = range(-10, -25, -1)
    
    loss = 0
    accuracy = 0
    n=0
    for text, label in zip(test_data, labels):
        predictions = experiment.classify(text, layers, split_token=".")

        # most_dishonest = np.argmax(predictions)
        # try:
        #     accuracy += label[most_dishonest] == 0
        # except:
        #     print(text, most_dishonest, label, predictions)
        probabilities = [1 - p/(p+1) for p in predictions] # probability statement is true
        print(probabilities)
        loss += sum([CE_loss(p, l) for p, l in zip(probabilities, label)])
        accuracy += sum([1 for p, l in zip(probabilities, label) if np.round(p) == l])
        n += len(predictions)

    print(f"Accuracy: {accuracy} out of {n}, average CE loss: {loss/n}")
    new_row = pd.DataFrame([{'Reader': reader, 'Dataset': dataset, 'Accuracy': accuracy}])
    results = pd.concat([results, new_row], ignore_index=True)

results.to_csv('./data/per_statement_preds.csv', index=False)
print(results)

