from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1, ConfusionMatrix
import pandas as pd

def init_metrics(device, num_classes):
    return MetricCollection({
        'micro_acc': Accuracy(num_classes=num_classes),
        'macro_acc': Accuracy(num_classes=num_classes, average='macro'),
        'macro_prec': Precision(num_classes=num_classes, average='macro'),
        'macro_rec': Recall(num_classes=num_classes, average='macro'),
        'macro_f1': Recall(num_classes=num_classes, average='macro'),
        'acc': Accuracy(num_classes=num_classes, average='none'),
        'prec': Precision(num_classes=num_classes, average='none'),
        'rec': Recall(num_classes=num_classes, average='none'),
        'f1': Recall(num_classes=num_classes, average='none'),
        'confusion_matrix': ConfusionMatrix(num_classes=num_classes)
    }).to(device)

def generate_macro_metrics_dataframe(metric_computed, loss):
    metric_table = {
        "metric": [
            "Loss",
            "Micro Accuracy", 
            "Macro Accuracy", 
            "Macro Precision", 
            "Macro Recall", 
            "Macro F1"
        ],
        "value": [
            loss,
            metric_computed["micro_acc"].item(), 
            metric_computed["macro_acc"].item(), 
            metric_computed["macro_prec"].item(), 
            metric_computed["macro_rec"].item(), 
            metric_computed["macro_f1"].item()
        ]
    }
    
    return pd.DataFrame(metric_table)

def generate_macro_metrics(metric_computed, loss, should_print=True):
    macro_metrics = generate_macro_metrics_dataframe(metric_computed, loss)
    
    if should_print:
        print('-'*25 + ' GENERAL METRICS ' + '-'*25)
        print()
        macro_metrics.apply(lambda x: print('{} {}'.format(x["metric"], x["value"])), axis=1)    
        print()
    
    return macro_metrics

def generate_per_class_dataframe(metric_computed, classes):
    metric_table = {"class": [], "accuracy":[], "precision": [], "recall": [], "f1": []}
    for (i, class_name) in enumerate(classes):
        metric_table["class"].append(class_name)
        metric_table["accuracy"].append(metric_computed["acc"][i].item())
        metric_table["precision"].append(metric_computed["prec"][i].item())
        metric_table["recall"].append(metric_computed["rec"][i].item())
        metric_table["f1"].append(metric_computed["f1"][i].item())
    return pd.DataFrame(metric_table)

def generate_per_class_metrics(metric_computed, classes, should_print=True):
    per_class_metrics = generate_per_class_dataframe(metric_computed, classes)
    if should_print:
        print('-'*25 + ' PER CLASS METRICS ' + '-'*25)
        display(per_class_metrics)
    return per_class_metrics
    
def record_macro_metrics(writer, phase, epoch, metric_computed, epoch_loss):
    writer.add_scalar('{}/Loss'.format(phase), epoch_loss, epoch)
    writer.add_scalar('{}/MicroAccuracy'.format(phase), metric_computed["micro_acc"], epoch)
    writer.add_scalar('{}/Accuracy'.format(phase), metric_computed["macro_acc"], epoch)
    writer.add_scalar('{}/Precision'.format(phase), metric_computed["macro_prec"], epoch)
    writer.add_scalar('{}/Recall'.format(phase), metric_computed["macro_rec"], epoch)
    writer.add_scalar('{}/F1'.format(phase), metric_computed["macro_f1"], epoch)
    writer.flush()