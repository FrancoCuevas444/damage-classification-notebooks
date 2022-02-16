import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1, ConfusionMatrix
import pandas as pd
import sklearn
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import os

import importlib
import training_helper
import dataset_modules.preloaded_one_part_dataset as opd
import metrics_helper

import dataset_modules.image_preloader as image_preloader
import evaluation.evaluation_helper as eh

def train_one_part_model(
    part, 
    preloaded_images, 
    model_name, 
    experiment_name, 
    device, 
    feature_extraction=False,
    train_ratio=0.8,
    random_state=42,
    num_epochs=25,
    ignore_repair=False,
    remove_repair=False,
    remove_not_visible=False,
    class_weights=None,
    visibility_file=None):
    
    os.makedirs("./trained_models/{}/{}/".format(model_name, experiment_name), exist_ok=True)
    
    dataset = opd.PreloadedOnePartDataset(
        part,
        preloaded_images,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        ignore_repair=ignore_repair,
        remove_repair=remove_repair,
        visibility_file=visibility_file,
        remove_not_visible=remove_not_visible
    )

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(random_state))

    classes = dataset.classes

    dataset_sizes = {
        'train': len(train_dataset),
        'test': len(test_dataset)
    }

    print("#TRAIN {} #TEST {}".format(dataset_sizes["train"], dataset_sizes["test"]))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    dataloaders = {
        'train': train_loader,
        'test': test_loader
    }
    
    model = models.resnet50(pretrained=True)
    NUM_CLASSES = len(classes)

    if feature_extraction:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = torch.nn.Linear(2048, NUM_CLASSES)
    model = model.to(device)

    # Tensorboard metrics writer
    writer = SummaryWriter(log_dir='./trained_models/{}/tensorboard/{}'.format(
        model_name, 
        experiment_name + '-' + datetime.now().strftime("%Y%m%d-%H%M%S")))

    # Función de error
    criterion = F.cross_entropy
    if class_weights is not None:
        class_weights = class_weights.to(device)

    # Optimizador
    parameters_to_update = model.parameters()

    if feature_extraction:
        parameters_to_update = model.fc.parameters()

    optimizer = optim.SGD(parameters_to_update, lr=0.001)
    
    model = training_helper.train_model(
        model, 
        criterion, 
        optimizer, 
        dataloaders, 
        dataset_sizes, 
        device, 
        writer, 
        NUM_CLASSES,
        'trained_models/{}/{}'.format(model_name, experiment_name),
        class_weights=class_weights,
        main_metric='macro_f1', 
        num_epochs=num_epochs,
        patience=5
    )
    
    best_model_path = './trained_models/{}/{}/best_model.pth'.format(model_name, experiment_name)
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    torch.save(model.state_dict(), best_model_path)
    
    #model.load_state_dict(torch.load(best_model_path))
    eh.evaluate_model(model_name, experiment_name, model, criterion, dataset_sizes, test_loader, classes, device, writer)
    
def train_one_part_model_dataugment(
    part, 
    preloaded_images, 
    model_name, 
    experiment_name, 
    device, 
    feature_extraction=False,
    train_ratio=0.8,
    random_state=42,
    num_epochs=25,
    ignore_repair=False,
    remove_not_visible=False,
    visibility_file=None,
    class_weights=None,
    data_augmentation=transforms.Compose([]),
    class_to_augment=None):
    
    os.makedirs("./trained_models/{}/{}/".format(model_name, experiment_name), exist_ok=True)
    
    print("LOAD TRAIN")
    train_dataset = opd.PreloadedOnePartDataset(
        part,
        preloaded_images,
        data_augmentation = data_augmentation,
        class_to_augment=class_to_augment,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        state_file="./dataset_modules/state-train.json",
        ignore_repair=ignore_repair,
        remove_not_visible=remove_not_visible,
        visibility_file=visibility_file
    )
    
    print("LOAD TEST")
    test_dataset = opd.PreloadedOnePartDataset(
        part,
        preloaded_images,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        state_file="./dataset_modules/state-test.json",
        ignore_repair=ignore_repair,
        remove_not_visible=remove_not_visible,
        visibility_file=visibility_file
    )

    train_size = len(train_dataset)
    test_size = len(test_dataset)
    classes = train_dataset.classes

    dataset_sizes = {
        'train': train_size,
        'test': test_size
    }

    print("#TRAIN {} #TEST {}".format(dataset_sizes["train"], dataset_sizes["test"]))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    dataloaders = {
        'train': train_loader,
        'test': test_loader
    }
    
    model = models.resnet50(pretrained=True)
    NUM_CLASSES = len(classes)

    if feature_extraction:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = torch.nn.Linear(2048, NUM_CLASSES)
    model = model.to(device)

    # Tensorboard metrics writer
    writer = SummaryWriter(log_dir='./trained_models/{}/tensorboard/{}'.format(
        model_name, 
        experiment_name + '-' + datetime.now().strftime("%Y%m%d-%H%M%S")))

    # Función de error
    criterion = F.cross_entropy
    if class_weights is not None:
        class_weights = class_weights.to(device)

    # Optimizador
    parameters_to_update = model.parameters()

    if feature_extraction:
        parameters_to_update = model.fc.parameters()

    optimizer = optim.SGD(parameters_to_update, lr=0.001)
    
    model = training_helper.train_model(
        model, 
        criterion, 
        optimizer, 
        dataloaders, 
        dataset_sizes, 
        device, 
        writer, 
        NUM_CLASSES,
        'trained_models/{}/{}'.format(model_name, experiment_name),
        class_weights=class_weights,
        main_metric='macro_f1', 
        num_epochs=num_epochs,
        patience=5
    )
    
    best_model_path = './trained_models/{}/{}/best_model.pth'.format(model_name, experiment_name)
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    torch.save(model.state_dict(), best_model_path)
    
    #model.load_state_dict(torch.load(best_model_path))
    eh.evaluate_model(model_name, experiment_name, model, criterion, dataset_sizes, test_loader, classes, device, writer)