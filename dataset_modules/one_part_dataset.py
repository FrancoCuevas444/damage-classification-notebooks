import pandas as pd
from torchvision.datasets import VisionDataset
from PIL import Image
from torchvision import transforms
import dataset_modules.common as common
import json
    
def one_part_classes(part):
    classes = [x.format(part.lower().replace(" ", "_")) for x in ["{}_roto", "{}_sano", "no_{}"]]
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def is_useful(row):
    return row["useful"] == "yes" and not(row["angle"] in [None, "other", ""])

class OnePartDataset(VisionDataset):
    """
        Clase encargada de computar un dataset que incluye tres classes:
            - tiene la parte sana
            - tiene la parte rota
            - no tiene la parte
    """

    def __init__(self, part, root='./dataset_modules/imgs/', is_useful=is_useful, state_file="./dataset_modules/state.json", complaint_parts="./preprocessing/piezas_normalizadas.csv", transform=None, target_transform=None, preload=False, ignore_repair_hours_greater_than=None):
        super(OnePartDataset, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        
        # Load metadata
        metadata = common.load_metadata_dataframe(state_file)
        self.metadata = metadata[metadata.apply(is_useful, axis=1)]
        self.complaint_parts = pd.read_csv(complaint_parts)
        self.complaint_parts = self.complaint_parts[self.complaint_parts["Tarea"] != "SYC"]
        if ignore_repair_hours_greater_than:
            self.complaint_parts = self.complaint_parts[~((self.complaint_parts["Tarea"] == "Reparar")&(self.complaint_parts["Horas"].astype(float) >= ignore_repair_hours_greater_than))]
        
        # Load classes
        classes, class_to_idx = one_part_classes(part)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.part = part
        self.samples = self.generate_samples()
        self.loader = common.pil_loader
        self.preload = preload
        
        if preload:
            self.preloaded_samples = [self.load_sample(path, target) for (path, target) in self.samples]
        
        common.print_class_distribution(classes, self.samples)
        

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        if self.preload:
            return self.preloaded_samples[index]
        
        path, target = self.samples[index]
        return self.load_sample(path, target)
    
    def load_sample(self, path, target):
        sample = self.loader(self.root + path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (sample, target, path)
    
    def generate_samples(self):
        return self.metadata.apply(lambda x: (x["image"], self.get_part_category(x)), axis=1).tolist()
        
    def get_part_category(self, row):
        is_visible = self.part in common.angulo_pieza[row["angle"]]
        is_broken = self.part in self.parts_from_complaint(row["image"].split("/")[0])

        if is_visible:
            if is_broken:
                return 0
            else:
                return 1
        else:
            return 2
    
    def parts_from_complaint(self, complaint):
        df_complaint = self.complaint_parts[self.complaint_parts["DENUNCIA"] == complaint]
        return df_complaint["pieza_normalizada"].tolist()