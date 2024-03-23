from pathlib import Path
import pandas as pd
from typing import List, Tuple


class NTruthMLieLoader:
    def __init__(self, m: int, n: int, path: Path):
        """initialize N truths M lies data loader

        Args:
            m (int): number of truth statements
            n (int): number of lie statements
            path (Path): path to the dataset (csv file format)
        """
        if m < 0 or n < 0 or m == n == 0:
            raise ValueError("m and n should be nonnegative integers; at least one must be positive")
        self.m = m
        self.n = n
        self.path = path

        self._load()

    def __len__(self) -> int:
        return len(self.df_true) + len(self.df_false)

    def _load(self) -> None:
        """load the dataset from self.path"""
        df = pd.read_csv(self.path)
        self.df_true = df.loc[df["label"] == 1]
        self.df_false = df.loc[df["label"] == 0]

    def sample_w_r(self, k: int) -> List[List[Tuple]]:
        """sample k rows from the dataset with replacement

        Args:
            k (int): number of datapoints to sample

        Returns:
            List[List[Tuple]]: list of k datapoints, each containing m truths and n lies in random order
        """
        return self._sample(k, replace=True)

    def sample_wo_r(self, k: int) -> List[List[Tuple]]:
        """sample k rows from the dataset without replacement

        Args:
            k (int): number of datapoints to sample. If k = -1, it will return all datapoints until it runs out

        Returns:
            List[List[Tuple]]: list of k datapoints, each containing m truths and n lies in random order
        """
        return self._sample(k, replace=False)

    def _sample(self, k: int, replace: bool) -> List[List[Tuple]]:
        """helper function for sampling

        Args:
            k (int): number of datapoints to sample
            replace (bool): whether to perform sampling with replacement or without replacement

        Raises:
            ValueError: if k = -1 and replace is set to True

        Returns:
            List[List[Tuple]]: list of k datapoints, each containing m truths and n lies in random order
        """
        if k == -1 and replace:
            raise ValueError("k cannot be -1 if sampling with replacement")
        elif k == -1 and not replace:
            if self.n and not self.m:
                k = len(self.df_false) // self.n
            elif self.m and not self.n:
                k = len(self.df_true) // self.m
            else:
                k = min(len(self.df_true) // self.m, len(self.df_false) // self.n)

        true_statements = self.df_true.sample(self.m * k, replace=replace)
        false_statements = self.df_false.sample(self.n * k, replace=replace)

        all_datapoints = [
            list(
                pd.concat(
                    [
                        true_statements.sample(self.m, replace=False),
                        false_statements.sample(self.n, replace=False),
                    ]
                )
                .sample(frac=1)  # randomize the order
                .itertuples(index=False, name=None)
            )
            for _ in range(k)
        ]

        return all_datapoints


if __name__ == "__main__":
    loader = NTruthMLieLoader(2, 1, Path("../data/facts_true_false.csv"))
    print(loader.sample_w_r(4))
