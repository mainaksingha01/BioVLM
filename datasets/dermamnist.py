# import os

# from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
# from dassl.utils import listdir_nohidden
# from .oxford_pets import OxfordPets

# @DATASET_REGISTRY.register()
# class DermaMNIST(DatasetBase):
#     """DermaMNIST dataset."""

#     dataset_dir = "dermamnist"

#     def __init__(self, cfg):
#         root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
#         self.dataset_dir = os.path.join(root, self.dataset_dir)

#         classnames = self.read_classnames()

#         train = self.read_data("train", classnames)
#         val = self.read_data("val", classnames)
#         test = self.read_data("test", classnames)
        
#         subsample = cfg.DATASET.SUBSAMPLE_CLASSES
#         train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

#         super().__init__(train_x=train, val=val, test=test)

#     def read_classnames(self):
#         # You might need to adjust this part based on how class names are stored
#         return {str(i): f"Class_{i}" for i in range(7)}  # DermaMNIST has 7 classes

#     def read_data(self, split, classnames):
#         split_dir = os.path.join(self.dataset_dir, split)
#         folders = listdir_nohidden(split_dir, sort=True)
#         items = []

#         for label in folders:
#             classname = classnames[label]
#             img_dir = os.path.join(split_dir, label)
#             imnames = listdir_nohidden(img_dir)
            
#             for imname in imnames:
#                 impath = os.path.join(img_dir, imname)
#                 item = Datum(
#                     impath=impath,
#                     label=int(label),
#                     classname=classname
#                 )
#                 items.append(item)

#         return items



import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing
import math

# from .oxford_pets import OxfordPets
# from .dtd import DescribableTextures as DTD


@DATASET_REGISTRY.register()
class DermaMNIST(DatasetBase):

    dataset_dir = "dermamnist"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.image_dir = os.path.join(root, self.dataset_dir)
        self.split_path = os.path.join(self.image_dir, "split_dermamnist.json")
        self.split_fewshot_dir = os.path.join(self.image_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        #if os.path.exists(self.split_path):
        train, val, test = self.read_split(self.split_path, self.image_dir)
        # else:
        #     train, val, test = DTD.read_and_split_data(self.image_dir, ignored=IGNORED, new_cnames=NEW_CNAMES)
        #     OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = self.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)
        
    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test
    
    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args
        
        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        else:
            selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}
        
        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)
        
        return output
