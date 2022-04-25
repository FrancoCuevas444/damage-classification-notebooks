import json
import pandas as pd
from PIL import Image

piezas = [
    'Guardabarro Delantero Derecho',
    'Guardabarro Delantero Izquierdo',
    'Guardabarro Trasero Derecho',
    'Guardabarro Trasero Izquierdo',
    'Puerta Delantera Derecha',
    'Puerta Delantera Izquierda',
    'Puerta Trasera Derecha',
    'Puerta Trasera Izquierda',
    'Panel Trasero',
    'Capot',
    'Tapa de Baul',
    'Zocalo Derecho',
    'Zocalo Izquierdo',
    'Farol Delantero Derecho',
    'Farol Delantero Izquierdo',
    'Farol Trasero Derecho',
    'Farol Trasero Izquierdo',
    'Techo',
    'Lateral de Caja Derecho',
    'Lateral de Caja Izquierdo',
    'Paragolpe Delantero',
    'Paragolpe Trasero'
]

angulo_pieza = {
    "frente": list(set([piezas[9], piezas[13], piezas[14], piezas[20]])),
    "atras": list(set([piezas[8], piezas[10], piezas[15], piezas[16], piezas[21]])),
    "lado_cond": list(set([piezas[1], piezas[3], piezas[5], piezas[7], piezas[12], piezas[14], piezas[16], piezas[19]])), # sumar paragolpes y capot?? 
    "lado_acomp": list(set([piezas[0], piezas[2], piezas[4], piezas[6], piezas[11], piezas[13], piezas[15], piezas[18]])),
    "frente_cond": list(set([
        piezas[9], piezas[14], piezas[20], # frente
        piezas[1], piezas[3], piezas[5], piezas[7], piezas[12], piezas[14], piezas[19] # lado conductor
    ])),
    "frente_acomp": list(set([
        piezas[9], piezas[13], piezas[20], # frente
        piezas[0], piezas[2], piezas[4], piezas[6], piezas[11], piezas[13], piezas[18] # lado acompañante
    ])),
    "atras_cond": list(set([
        piezas[8], piezas[10], piezas[16], piezas[21], # atras
        piezas[1], piezas[3], piezas[5], piezas[7], piezas[12], piezas[16], piezas[19] # lado conductor
    ])),
    "atras_acomp": list(set([
        piezas[8], piezas[10], piezas[15], piezas[21], # atras
        piezas[0], piezas[2], piezas[4], piezas[6], piezas[11], piezas[15], piezas[18] # lado acompañante
    ]))
}

def print_class_distribution(classes, samples):
    print("----- CLASS DISTRIBUTION -----")
    classes_distribution = { x: 0 for x in classes }
    for elem in samples:
        classn = elem[1]
        classes_distribution[classes[classn]] += 1

    for elem in sorted(classes_distribution.items(), key=lambda item: item[1], reverse=True):
        print("Class: {}, #{}, {:.2f}%".format(elem[0], elem[1], elem[1]/len(samples)*100))
        
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def is_useful(row):
    return row["useful"] == "yes" and not(row["angle"] in [None, "other", ""])
    
def load_metadata_dataframe(state_file, filter_useful=True):
    with open(state_file) as f:
        state = json.load(f)
    
    data_for_dataframe = {"image": [], "useful": [], "angle": []}

    for (k,v) in state.items():
        data_for_dataframe["image"].append(k)
        data_for_dataframe["useful"].append(v["useful"])
        data_for_dataframe["angle"].append(v["photo_angle"])
    
    df = pd.DataFrame(data_for_dataframe)
    
    if filter_useful:
        df = df[df.apply(is_useful, axis=1)]
        
    return df

def load_complaint_parts(complaint_parts_file, ignore_repair, ignore_repair_hours_greater_than, ignore_syc=True):
    complaint_parts = pd.read_csv(complaint_parts_file)
    
    # filtar "SYC" (sacar y colocar)
    if ignore_syc:
        complaint_parts = complaint_parts[complaint_parts["Tarea"] != "SYC"]
        
    if ignore_repair:
        complaint_parts = complaint_parts[complaint_parts["Tarea"] != "Reparar"]

    if ignore_repair_hours_greater_than:
        complaint_parts = complaint_parts[~((complaint_parts["Tarea"] == "Reparar")&(complaint_parts["Horas"].astype(float) >= ignore_repair_hours_greater_than))]
        
    return complaint_parts