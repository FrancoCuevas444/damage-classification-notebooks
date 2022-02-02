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

class OnePartDataset(VisionDataset):
    """
        Clase encargada de computar un dataset que incluye tres classes:
            - tiene la parte sana
            - tiene la parte rota
            - no tiene la parte
    """

    def __init__(self, 
                 part, 
                 root='./dataset_modules/imgs/', 
                 state_file="./dataset_modules/state.json", 
                 complaint_parts_file="./preprocessing/piezas_normalizadas.csv", 
                 transform=None, 
                 target_transform=None, 
                 preload=False, 
                 ignore_repair=False, 
                 ignore_repair_hours_greater_than=None, 
                 visibility_file=None
                ):
        
        super(OnePartDataset, self).__init__(root, 
                                             transform=transform,
                                            target_transform=target_transform)
        
        # Load metadata
        self.metadata = common.load_metadata_dataframe(state_file, filter_useful=True)
        
        # Load complaint parts
        self.complaint_parts = common.load_complaint_parts(complaint_parts_file, ignore_repair, ignore_repair_hours_greater_than)
        
        # Load visibility information if present
        self.visibility = None
        if visibility_file:
            self.visibility = pd.read_csv(visibility_file)
            self.visibility.set_index("img")
        
        # Load classes
        classes, class_to_idx = one_part_classes(part)
        self.classes = classes
        self.class_to_idx = class_to_idx
        
        self.part = part
        self.samples = self.generate_samples()
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
        sample = common.pil_loader(self.root + path)
        
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
        
        is_damage_not_visible = False
        try:
            if self.visibility is not None:
                is_damage_not_visible = self.visibility.loc[self.visibility["img"] == row["image"]]["visible_damage"].item() == "not_visible"
        except ValueError:
            is_damage_not_visible = False
        
        if is_visible:
            if is_broken and not is_damage_not_visible:
                return 0
            else:
                return 1
        else:
            return 2
    
    def parts_from_complaint(self, complaint):
        df_complaint = self.complaint_parts[self.complaint_parts["DENUNCIA"] == complaint]
        return df_complaint["pieza_normalizada"].tolist()