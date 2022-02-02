from dataset_modules.common import pil_loader
import torch
import torchvision.transforms as transforms
import metrics_helper
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_model(model_name, exp_name, model, criterion, dataset_sizes, test_loader, classes, device, writer):
    model.eval()
    
    metrics = metrics_helper.init_metrics(device, len(classes))
    tensorboard_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    total_loss = 0.0

    for i, (images, labels, imgs_path) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            metrics(outputs, labels)

            predictions = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            for sampleno in range(images.shape[0]):
                if(labels[sampleno] != predictions[sampleno]):
                    name = 'Misclasified_Predicted-{}_Classified-{}/{}'.format(
                        classes[predictions[sampleno]], 
                        classes[labels[sampleno]], 
                        imgs_path[sampleno])

                    writer.add_image(
                        name, 
                        tensorboard_transforms(pil_loader('./dataset_modules/imgs/' + imgs_path[sampleno]))
                    )
                    writer.flush()

    total_loss /= dataset_sizes["test"]

    metrics_result = metrics.compute()
    
    macro_metrics = metrics_helper.generate_macro_metrics(metrics_result, total_loss)
    per_class_metrics = metrics_helper.generate_per_class_metrics(metrics_result, classes)

    macro_metrics.to_csv('./trained_models/{}/{}/best_model_macro_metrics.csv'.format(model_name, exp_name), index=False)
    per_class_metrics.to_csv('./trained_models/{}/{}/best_model_per_class_metrics.csv'.format(model_name, exp_name), index=False)
    
    df_cm = pd.DataFrame(
        metrics_result['confusion_matrix'], 
        index = classes,
        columns = classes)

    df_cm = df_cm.applymap(lambda x: x.item())

    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, cmap="Blues")
    df_cm.to_csv('./trained_models/{}/{}/confusion_matrix.csv'.format(model_name, exp_name))