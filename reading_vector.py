from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch
import numpy as np
import numpy as np
import seaborn as sns
import pandas as pd
from honesty.utils import honesty_function_dataset, plot_lat_scans, plot_detection_results
from repe import repe_pipeline_registry, PCARepReader, ClusterMeanRepReader
repe_pipeline_registry()

from honesty.utils import honesty_function_dataset, plot_lat_scans, plot_detection_results



class NTruthMLieExperiment():

    def __init__(self, model_name_or_path, 
                 device = "auto") -> None:
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map=device, 
                                                    offload_state_dict=True, offload_folder = "offload")
        use_fast_tokenizer = "LlamaForCausalLM" not in self.model.config.architectures
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
        self.tokenizer.pad_token_id = 0
        self.rep_reading_pipeline =  pipeline("rep-reading", model=self.model, tokenizer=self.tokenizer)
        self.hidden_layers = list(range(-1, -self.model.config.num_hidden_layers, -1))
        self.reader = None
        self.user_tag = "[INST]"
        self.assistant_tag = "[/INST]"


    def load_rep_reader(self, filename, reader_type="pca"):
        if reader_type == 'pca':
            ReaderClass = PCARepReader
        elif reader_type == 'cluster':
            ReaderClass = ClusterMeanRepReader
        honesty_rep_reader = ReaderClass.from_pretrained(filename)
        self.reader = honesty_rep_reader
        return honesty_rep_reader

    def save_rep_reader(self):
        self.reader.save(self.model_name_or_path, self.dataset_name)
        return None

    def train_rep_reader(self, dataset, labels, rep_token=-1, 
                         n_difference=1, direction_method='pca', batch_size=32):

        honesty_rep_reader = self.rep_reading_pipeline.get_directions(
            dataset, 
            rep_token=rep_token, 
            hidden_layers=self.hidden_layers, 
            n_difference=n_difference, 
            train_labels=labels, 
            direction_method=direction_method,
            batch_size=batch_size,
        )
        self.reader = honesty_rep_reader
        return honesty_rep_reader

    def get_honesty_scores(self, data, rep_token=-1):
        return self.rep_reading_pipeline(
            data, 
            hidden_layers=self.hidden_layers,
            rep_token=rep_token,
            rep_reader=self.reader,
            batch_size=32)


    def accuracy_by_layer(self, dataset, savename):
        plt.figure() 
        test_scores = self.get_honesty_scores(dataset, self.hidden_layers, rep_token=-1)
        
        results = {layer: {} for layer in self.hidden_layers}
        rep_readers_means = {}
        rep_readers_means['honesty'] = {layer: 0 for layer in self.hidden_layers}

        for layer in self.hidden_layers:
            H_test = [H[layer] for H in test_scores]
            rep_readers_means['honesty'][layer] = np.mean(H_test)
            H_test = [H_test[i:i+2] for i in range(0, len(H_test), 2)]
            
            sign = self.reader.direction_signs[layer]

            eval_func = min if sign == -1 else max
            cors = np.mean([eval_func(H) == H[0] for H in H_test])
            
            results[layer] = cors

        plt.plot(self.hidden_layers, [results[layer] for layer in self.hidden_layers])
        plt.savefig(f"./results/{savename}.png")

    def get_model_response(self, prompts):
        
        template_str = '{user_tag} {scenario} {assistant_tag}'
        test_input = [template_str.format(scenario=s, user_tag=self.user_tag, assistant_tag=self.assistant_tag) for s in prompts]

        test_data = []
        for t in test_input:
            with torch.no_grad():
                output = self.model.generate(**self.tokenizer(t, return_tensors='pt').to(self.model.device), max_new_tokens=30)
            completion = self.tokenizer.decode(output[0], skip_special_tokens=True)
            test_data.append(completion)
        return test_data


    def honesty_detection(self, test_data, layers, threshold=0.0, start_answer_token=None, savename=None):

        if start_answer_token is None:
            start_answer_token = self.tokenizer.tokenize(self.assistant_tag)[-1]
        
        ans = []

        for (i, chosen_str) in enumerate(test_data): 
            rep_reader_scores_dict = {}
            rep_reader_scores_mean_dict = {}
            input_ids = self.tokenizer.tokenize(chosen_str)
            results = []
            for ice_pos in range(len(input_ids)):
                ice_pos = -len(input_ids) + ice_pos
                H_tests = self.rep_reading_pipeline([chosen_str],
                                            rep_reader=self.reader,
                                            rep_token=ice_pos,
                                            hidden_layers=self.hidden_layers)
                results.append(H_tests)

                honesty_scores = []
                honesty_scores_means = []
                for pos in range(len(results)):
                    tmp_scores = []
                    tmp_scores_all = []
                    for layer in self.hidden_layers:
                        tmp_scores_all.append(results[pos][0][layer][0] * self.reader.direction_signs[layer][0])
                        if layer in layers:
                            tmp_scores.append(results[pos][0][layer][0] * self.reader.direction_signs[layer][0])
                    honesty_scores.append(tmp_scores_all)
                    honesty_scores_means.append(np.mean(tmp_scores))

            rep_reader_scores_dict['honesty'] = honesty_scores
            rep_reader_scores_mean_dict['honesty'] = honesty_scores_means
            ans.append(honesty_scores_means)
            if savename is not None:
                fig, ax = plot_detection_results(input_ids, rep_reader_scores_mean_dict, threshold, start_answer_token=start_answer_token)
                fig.savefig(f"./results/{savename}_{i}.png")
        return ans
    
    def classify(self, text, layers, split_token = "."):
        """Classifies the honesty of a set of sentences.
        text (str): text to classify
        layers (list): list of layers to consider
        threshold (float): threshold for classification
        """
        
        results = self.honesty_detection([text], layers)
        tokens = self.tokenizer.tokenize(text)
        split_occurrences = [i+1 for i, token in enumerate(tokens) if token == split_token]
        predictions = []
        for (i, j) in zip([0]+[k+1 for k in split_occurrences[:-1]], split_occurrences):
            dishonest_scores = [x for x in results[0][i:j] if x < 0]
            predictions.append(-sum(dishonest_scores)/(j-i+1))

        return predictions


    
        
def compare_directions(v1, v2, s1, s2):
    """Compares the directions of two sets of vectors v1 and v2, with corresponding signs s1 and s2.
    v1 (dict): {layer: vector}
    v2 (dict): {layer: vector}
    s1 (dict): {layer: sign}
    s2 (dict): {layer: sign}
    """
    return {i : (s1[i][0]*s2[i][0])*np.dot(v1[i][0], v2[i][0]) / (np.linalg.norm(v1[i]) * np.linalg.norm(v2[i])) for i in v1.keys()}


if __name__ == "__main__":

    
    model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
    experiment = NTruthMLieExperiment(model_name_or_path)


    experiment.load_rep_reader("mistral_MM_facts_true_false", "cluster")

    true_then_false = ["[/INST] The city of Malang is in Indonesia. The city of Lahore is in China. "]
    layers = range(-10, -25, -1)
    experiment.honesty_detection(true_then_false, layers, threshold=0.0, savename="true_then_false")
    