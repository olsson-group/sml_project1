from collections import defaultdict

import matplotlib.pyplot as plt
import numpy
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sml.scaffold_splitter import ScaffoldSplitter
from torch.utils.data import Subset


def split_dataset(dataset, frac_train, seed=None, split=None):
    assert (
        seed is not None
    ), "please provide a seed for the splitting for reproducibility"

    assert split in [
        "random",
        "scaffold",
    ], "split must be either 'random' or 'scaffold' random split provides a completely random split of all molecules in the datasret, while scaffold split provides a split based on the scaffold of the molecules in the dataset. Scaffold splitting is useful when you want to split the dataset based on the chemical similarity of the molecules in the dataset, but note that splitting by scaffold validation data may or may not include 'easier' molecules that the training data."

    torch.manual_seed(seed)

    if split == "random":

        len_train = int(len(dataset) * frac_train)
        len_val = len(dataset) - len_train

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [len_train, len_val]
        )
        return train_dataset, val_dataset

    elif split == "scaffold":
        scaffold_splitter = ScaffoldSplitter()
        smiles_list = [data.smiles for data in dataset]

        frac_valid = 1 - frac_train

        train_idxs, valid_idxs = scaffold_splitter.train_valid_split(
            dataset, smiles_list, frac_train=frac_train, frac_valid=frac_valid
        )

        train_dataset = Subset(dataset, train_idxs)
        val_dataset = Subset(dataset, valid_idxs)

        return train_dataset, val_dataset


def filter_dataset(dataset):
    valid_idxs = []
    for idx, data in enumerate(dataset):
        if data.x.shape[0] <= 2:
            continue
        valid_idxs.append(idx)
    return Subset(dataset, valid_idxs)


def color_cycle():
    for c in plt.rcParams["axes.prop_cycle"]:
        yield c["color"]


class BaseSplitter(object):
    def k_fold_split(self, dataset, k):
        raise NotImplementedError

    def _split(self, dataset, **kwargs):
        raise NotImplementedError

    def train_valid_test_split(
        self, dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, **kwargs
    ):

        train_inds, valid_inds, test_inds = self._split(
            dataset, frac_train, frac_valid, frac_test, **kwargs
        )

        return train_inds, valid_inds, test_inds

    def train_valid_split(self, dataset, frac_train=0.9, frac_valid=0.1, **kwargs):

        train_inds, valid_inds, test_inds = self._split(
            dataset, frac_train, frac_valid, 0.0, **kwargs
        )
        assert len(test_inds) == 0

        return train_inds, valid_inds


def generate_scaffold(smiles, include_chirality=False):
    """return scaffold string of target molecule"""
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality
    )
    return scaffold


class ScaffoldSplitter(BaseSplitter):
    """Class for doing data splits by chemical scaffold.

    Referred Deepchem for the implementation, https://git.io/fXzF4
    """

    def _split(self, dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, **kwargs):
        numpy.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        seed = kwargs.get("seed", None)
        smiles_list = kwargs.get("smiles_list")
        include_chirality = kwargs.get("include_chirality")
        if len(dataset) != len(smiles_list):
            raise ValueError("The lengths of dataset and smiles_list are " "different")

        rng = numpy.random.RandomState(seed)

        scaffolds = defaultdict(list)
        for ind, smiles in enumerate(smiles_list):
            scaffold = generate_scaffold(smiles, include_chirality)
            scaffolds[scaffold].append(ind)

        scaffold_index = rng.permutation(numpy.arange(len(scaffolds.values())))
        scaffold_sets = [i for i in scaffolds.values()]

        n_total_valid = int(numpy.floor(frac_valid * len(dataset)))
        n_total_test = int(numpy.floor(frac_test * len(dataset)))

        train_index = []
        valid_index = []
        test_index = []

        for ssi in scaffold_index:
            if len(valid_index) + len(scaffold_sets[ssi]) <= n_total_valid:
                valid_index.extend(scaffold_sets[ssi])
            elif len(test_index) + len(scaffold_sets[ssi]) <= n_total_test:
                test_index.extend(scaffold_sets[ssi])
            else:
                train_index.extend(scaffold_sets[ssi])

        return (
            numpy.array(train_index),
            numpy.array(valid_index),
            numpy.array(test_index),
        )

    def train_valid_test_split(
        self,
        dataset,
        smiles_list,
        frac_train=0.8,
        frac_valid=0.1,
        frac_test=0.1,
        seed=None,
        include_chirality=False,
        **kwargs
    ):
        """Split dataset into train, valid and test set.

        Split indices are generated by splitting based on the scaffold of small
        molecules.

        Args:
            dataset(NumpyTupleDataset, numpy.ndarray):
                Dataset.
            smiles_list(list):
                SMILES list corresponding to datset.
            seed (int):
                Random seed.
            frac_train(float):
                Fraction of dataset put into training data.
            frac_valid(float):
                Fraction of dataset put into validation data.

        Returns:
            SplittedDataset(tuple): splitted dataset or indices

        """
        return super(ScaffoldSplitter, self).train_valid_test_split(
            dataset,
            frac_train,
            frac_valid,
            frac_test,
            seed=seed,
            smiles_list=smiles_list,
            include_chirality=include_chirality,
            **kwargs
        )

    def train_valid_split(
        self,
        dataset,
        smiles_list,
        frac_train=0.9,
        frac_valid=0.1,
        seed=None,
        include_chirality=False,
        **kwargs
    ):
        """Split dataset into train and valid set.

        Split indices are generated by splitting based on the scaffold of small
        molecules.

        Args:
            dataset(NumpyTupleDataset, numpy.ndarray):
                Dataset.
            smiles_list(list):
                SMILES list corresponding to datset.
            seed (int):
                Random seed.
            frac_train(float):
                Fraction of dataset put into training data.
            frac_valid(float):
                Fraction of dataset put into validation data.
            converter(callable):
            return_index(bool):
                If `True`, this function returns only indices. If `False`, this
                function returns splitted dataset.

        Returns:
            SplittedDataset(tuple): splitted dataset or indices

        """
        return super(ScaffoldSplitter, self).train_valid_split(
            dataset,
            frac_train,
            frac_valid,
            seed=seed,
            smiles_list=smiles_list,
            include_chirality=include_chirality,
            **kwargs
        )
