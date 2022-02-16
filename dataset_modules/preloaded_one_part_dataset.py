import pandas as pd
from torchvision.datasets import VisionDataset
from PIL import Image
from torchvision import transforms
import dataset_modules.common as common
import json
    
def one_part_classes(part, remove_not_visible):
    class_suffixes = ["{}_roto", "{}_sano", "no_{}"]
    if remove_not_visible:
        class_suffixes = ["{}_roto", "{}_sano"]

    classes = [x.format(part.lower().replace(" ", "_")) for x in class_suffixes]
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

class PreloadedOnePartDataset():
    """
        Clase encargada de computar un dataset que incluye tres classes:
            - tiene la parte sana
            - tiene la parte rota
            - no tiene la parte
    """

    def __init__(self, 
                 part,
                 preloaded_images,
                 state_file="./dataset_modules/state.json", 
                 complaint_parts_file="./preprocessing/piezas_normalizadas.csv", 
                 transform=None,
                 data_augmentation=None,
                 remove_not_visible=False,
                 class_to_augment=None,
                 target_transform=None, 
                 ignore_repair=False,
                 remove_repair=False,
                 ignore_repair_hours_greater_than=None, 
                 visibility_file=None
                ):
        
        self.part = part
        # Load metadata
        self.metadata = common.load_metadata_dataframe(state_file, filter_useful=True)
        
        # Load complaint parts
        self.complaint_parts = common.load_complaint_parts(complaint_parts_file, ignore_repair, ignore_repair_hours_greater_than)
        
        if remove_repair:
            self.remove_repair()
        
        # Load visibility information if present
        self.visibility = None
        if visibility_file:
            self.visibility = pd.read_csv(visibility_file)
            self.visibility.set_index("img")
        
        # Load classes
        classes, class_to_idx = one_part_classes(part, remove_not_visible)
        self.classes = classes
        self.class_to_idx = class_to_idx
        
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.class_to_augment = class_to_augment
        self.target_transform = target_transform
        self.preloaded_images = preloaded_images
        self.samples = self.generate_samples(remove_not_visible)
        
        common.print_class_distribution(classes, self.samples)
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_data, label, path = self.samples[index]
        transformed_label = self.transform_target(label)
        transformed_image_data = self.transform_img(image_data, transformed_label)
        return (transformed_image_data, transformed_label, path)
    
    def generate_samples(self, remove_not_visible):
        return self.metadata.apply(lambda x: self.generate_sample(x, remove_not_visible), axis=1).dropna().tolist()
    
    def generate_sample(self, row, remove_not_visible):
        is_visible = self.part in common.angulo_pieza[row["angle"]]
        if remove_not_visible and not is_visible:
            return
            
        path = row["image"]
        image_data = self.preloaded_images[row["image"]]
        label = self.get_part_category(row)
        return (image_data, label, path)
    
    def transform_img(self, img, label):
        transformed_img = img
        if (self.data_augmentation is not None) and (self.class_to_augment is None or self.class_to_augment == label):
            transformed_img = self.data_augmentation(img)
        
        if self.transform is not None:
            transformed_img = self.transform(img)

        return transformed_img
    
    def transform_target(self, target):
        transformed_target = target
        if self.target_transform is not None:
            transformed_target = self.target_transform(target)
        
        return transformed_target
    
    def remove_repair(self):
        print("remove repair only works with paragolpe del")
        self.metadata["DENUNCIA"] = self.metadata["image"].str.split("/").str[0]
        i1 = self.complaint_parts[(self.complaint_parts["pieza_normalizada"] == self.part) & (self.complaint_parts["Tarea"] == "Reparar")].set_index("DENUNCIA").index
        i2 = self.metadata.set_index("DENUNCIA").index
        print(len(i1))
        self.metadata = self.metadata[(~(self.metadata["angle"].isin(["frente", "frente_cond","frente_acomp"])))|(~(i2.isin(i1)))]
    
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