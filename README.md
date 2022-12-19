## Introducción

Este proyecto contiene el código referente a la experimentación con los modelos en el margen del proyecto de grado _"Tasación de siniestros de automóviles mediante reconocimiento de imágenes aplicando inteligencia computacional"_.

Este proyecto consistió en la construcción de modelos de aprendizaje automático para la detección del daño en vehículos en base a imágenes.

## Tecnologías utilizadas

- Python
- Pytorch
- Torchvision

## Organización del código

**Experimentos**

Los principales experimentos se encuentran en la carpeta raíz del repo y consisten en Jupyter Notebooks con el código asociado a los experimentos.

El archivo `main.py` contiene las funciones con la lógica utilizadas para el entrenamiento del modelo. Los archivos `metrics_helper.py` y `training_helper.py` contienen una serie de fucniones auxiliares para la obtención de métricas y el entrenamiento del modelo. 

**Módulos**

El módulo `dataset_modules` contiene toda la lógica de construcción del dataset. Esto implica la lectura de las imágenes y la asociación a cada imágen de la marca correspondiente para el conjunto de entrenamiento.

El módulo `evaluation` contiene código asociado a la evaluación del modelo. Esto implica la obtención de la métricas a través del archivo `evaluation_helper.py`.

El módulo `first_models` contiene los modelos legacy que fueron construidos en etapas tempranas del proyecto.

El módulo `interpretability` contiene una serie de notebooks con pruebas de concepto de herramientas de interpretabilidad.

El módulo `manual_labeling` contiene el código asociado a depurar la marca del dataset con etiquetado manual.

El módulo `preprocessing` continee el código asociado a parsear los archivos de metadata provistos por el BSE.

### Autores

Franco Cuevas, Ignacio Alonso, Lucas Barenchi.

**Supervisor:** Sergio Nesmanchow.