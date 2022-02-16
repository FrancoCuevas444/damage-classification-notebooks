import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1, ConfusionMatrix
import metrics_helper
import time
import copy
import pandas as pd
import numpy as np

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1
        print("ES Module Best: {}, BadEpochs: {}".format(self.best, self.num_bad_epochs))
        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

def train_model(
    model, 
    criterion, 
    optimizer, 
    dataloaders, 
    dataset_sizes, 
    device, 
    writer, 
    num_classes, 
    csv_folder,
    class_weights=None,
    main_metric="micro_acc", 
    num_epochs=10, 
    patience=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_main_metric = 0.0
    best_epoch_loss = 1.0
    best_epoch = 0
    
    metric_files = {
        'train': pd.DataFrame(),
        'test': pd.DataFrame()
    }
    
    es = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and test phase
        for phase in ['train', 'test']:
            phase_timer = time.time()
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            metrics = metrics_helper.init_metrics(device, num_classes)

            # Iterate over data.
            for inputs, labels, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels, weight=class_weights)
                    metrics(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            metrics_computed = metrics.compute()
            epoch_main_metric = metrics_computed[main_metric]

            print('{} Loss: {:.4f} Main metric ({}): {:.4f} Time: {:.2f}s'.format(
                phase, epoch_loss, main_metric, epoch_main_metric, time.time() - phase_timer))
            epoch_macro_metrics = metrics_helper.generate_macro_metrics(metrics_computed, epoch_loss, should_print=False)
            metrics_helper.record_macro_metrics(writer, phase, epoch, metrics_computed, epoch_loss)
            
            epoch_macro_metrics.insert(0, 'epoch', epoch)
            metric_files[phase] = metric_files[phase].append(epoch_macro_metrics)

            if phase == 'test':
                if epoch_main_metric > best_main_metric:
                    best_main_metric = epoch_main_metric
                
                if epoch_loss < best_epoch_loss:
                    best_epoch = epoch
                    best_epoch_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                
                if es.step(epoch_loss):
                    print("Early stopping... Best Epoch: {}".format(best_epoch))
                    break
        else:
            continue  # only executed if the inner loop did NOT break
        break

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test epoch loss: {:4f}'.format(best_epoch_loss))
    print('Best test main metric: {:4f}'.format(best_main_metric))
    
    metric_files['train'].to_csv('./{}/train_metrics.csv'.format(csv_folder), index=False)
    metric_files['test'].to_csv('./{}/test_metrics.csv'.format(csv_folder), index=False)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model