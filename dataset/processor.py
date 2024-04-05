from loader import NTruthMLieLoader
# from models.base import BaseModel
# from models.mistral import MistralModel
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import List, Union, Tuple
import torch
import csv
from tqdm import tqdm
from transformers import PreTrainedTokenizer

# class NTruthMLieProcessor:

#     def __init__(self, loader: NTruthMLieLoader, model: BaseModel):
#         self.loader = loader
#         self.model = model
#         self.layer_names = self.model.register_hook("act_fn", all=True)

#         self.dataset = NTruthMLieDataset(self.loader)

#     def process(self, save=True):
#         with open("./data/activations/association.csv", "w") as f:
#             writer = csv.writer(f, delimiter=",")
#             for i in tqdm(range(len(self.dataset))):
#                 sentences, labels = self.dataset[i]

#                 for sentence in sentences:
#                     self.model.infer(
#                         [
#                             {"role": "user", "content": ""},
#                             {"role": "assistant", "content": sentence},
#                         ],
#                         new_tokens=1,
#                     )

#                 all_tensors = [
#                     self.model.activations[layer_name]
#                     for layer_name in self.layer_names
#                 ]
#                 if save:
#                     torch.save(
#                         all_tensors,
#                         "./data/activations/" + str(i) + ".pt",
#                         )
#                 writer.writerow([str(i), sentences[-1], labels])
#                 self.model.activations.clear()
#                 return all_tensors


class NTruthMLieDataset(Dataset):
    def __init__(self, loader: NTruthMLieLoader):
        self.loader = loader
        self.data = self.loader.sample_wo_r(-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[List[str], str]:
        datapoint = self.data[idx]
        sentences = self._accumulate([sentence[0] for sentence in datapoint])
        labels = "".join([str(sentence[1]) for sentence in datapoint])

        return sentences, labels

    def _accumulate(
        self, sentences: Union[Tuple, List[str]], sep: str = " "
    ) -> List[str]:
        """accumulate the string in sentences such that if sentences = [A, B, C], the result is [A, AB, ABC]

        Args:
            sentences (Union[Tuple, List[str]]): list of strings to accumulate
            sep (str): separator to use when accumulating

        Returns:
            List[str]: accumulated list of strings
        """
        cur = ""
        accumulated = []
        for i, sentence in enumerate(sentences):
            cur += (sep if i > 0 else "") + sentence
            accumulated.append(cur)
        return accumulated

    def save(self, path: str):
        headers = ["statement", "label", "truth"]
        with open(path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for i in range(len(self.data)):
                sentences, labels = self[i]
                writer.writerow([sentences[-1], 1, labels])

        




if __name__ == "__main__":

    # processor = NTruthMLieProcessor(
    #     NTruthMLieLoader(2, 2, Path("./data/facts_true_false.csv")),
    #     MistralModel(torch.device("mps")),
    #     torch.device("mps"),
    # )
    # processor.process()
    # dataset_twotrue = NTruthMLieDataset(NTruthMLieLoader(2, 0, Path("./data/facts_true_false.csv")))
    dataset = NTruthMLieDataset(NTruthMLieLoader(3, 1, Path("./data/cities.csv")))

    dataset.save("./data/m=3n=1_cities.csv")
    # dataset_truefalse.save("./data/m=1n=1.csv")