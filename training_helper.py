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

def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, writer, num_classes, csv_folder, main_metric="micro_acc", num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_main_metric = 0.0
    
    metric_files = {
        'train': pd.DataFrame(),
        'test': pd.DataFrame()
    }

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
                    loss = criterion(outputs, labels)
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
            epoch_macro_metrics = metrics_helper.generate_macro_metrics(metrics_computed, epoch_loss)
            metrics_helper.record_macro_metrics(writer, phase, epoch, metrics_computed, epoch_loss)
            
            epoch_macro_metrics.insert(0, 'epoch', epoch)
            metric_files[phase] = metric_files[phase].append(epoch_macro_metrics)

            # deep copy the model
            if phase == 'test' and epoch_main_metric > best_main_metric:
                best_main_metric = epoch_main_metric
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test main metric: {:4f}'.format(best_main_metric))
    
    metric_files['train'].to_csv('./{}/train_metrics.csv'.format(csv_folder), index=False)
    metric_files['test'].to_csv('./{}/test_metrics.csv'.format(csv_folder), index=False)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model