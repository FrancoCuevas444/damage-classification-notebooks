import pandas as pd
from torchvision.datasets import VisionDataset
from PIL import Image
from torchvision import transforms
import dataset_modules.common as common
import json

def angle_classes(metadata):
    classes = sorted(metadata["angle"].unique().tolist())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def is_useful(row):
    return row["useful"] == "yes" and not(row["angle"] in [None, "other", ""])

class AngleDataset(VisionDataset):
    """
        Clase encargada de computar un dataset que incluye tres classes:
            - tiene la parte sana
            - tiene la parte rota
            - no tiene la parte
    """

    def __init__(self, root='./dataset_modules/imgs/', is_useful=is_useful, state_file="./dataset_modules/state.json", complaint_parts="./preprocessing/piezas_normalizadas.csv", transform=None, target_transform=None):
        super(AngleDataset, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        
        # Load metadata
        metadata = common.load_metadata_dataframe(state_file)
        self.metadata = metadata[metadata.apply(is_useful, axis=1)]
        
        # Load classes
        classes, class_to_idx = angle_classes(self.metadata)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = self.generate_samples()
        self.loader = common.pil_loader
        
        common.print_class_distribution(classes, self.samples)
        

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(self.root + path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (sample, target, path)
    
    def generate_samples(self):
        return self.metadata.apply(lambda x: (x["image"], self.class_to_idx[x["angle"]]), axis=1).tolist()
