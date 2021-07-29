import matplotlib
from IPython.core.interactiveshell import InteractiveShell
import seaborn as sns
# PyTorch
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
from torch.optim import lr_scheduler

from torchmetrics import Accuracy

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Data science tools
import numpy as np
import pandas as pd
import os
import json

# Image manipulations
from PIL import Image
# Useful for examining network
# from torchsummary import summary
# Timing utility
from timeit import default_timer as timer

# Visualizations
import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams['font.size'] = 14
from tqdm import tqdm

#custom class
from dataset import MyDataset
from slowfastnet import SlowFast,Bottleneck
from TrainTestCode import train

# Printing out all outputs
InteractiveShell.ast_node_interactivity = 'all'



# %%

# functions
def getClassDict(jsonfile):
    with open(jsonfile) as json_file:
        data = json.load(json_file)

        return data

def mapIntToClass(prediction,classjson):
    class_dict = getClassDict(classjson)
    keys_labels = list(class_dict.keys())
    values_int = list(class_dict.values())
    position = values_int.index(prediction)
    label = keys_labels[position]

    return label

def mapPrection2Class(y):
    class_dict = getClassDict("HACS_clips_v1.1__val_dictionary.json")
    vals = class_dict[y]
    path_label = vals[2]
    label = path_label.split("/")[6]

    return label

class UnNormalize(object):
    def __init__(self,mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def show_img():
    pass

def predict(dataLoader, model, jsonfile):

    model.eval()

    for batch_ind, batch in enumerate(dataLoader):
        # batch = batch.cuda()
        img, actual_y = batch
        y =  model(img)
        #top 5: print top-5 predictions
        post_act = torch.nn.Softmax(dim=1)
        preds = post_act(y)
        pred_classes = preds.topk(k=5).indices[0]
        # print(preds)
        pred_class_names = []#[mapPrection2Class(str(i)) for i in pred_classes]
        pred_probabilities = []
        for indx,i in enumerate(pred_classes):
            i = i.detach().numpy()
            pred_class_names.append(mapIntToClass(int(i),jsonfile))
            probs = preds[0][pred_classes[indx]]
            # print(probs.item())
            pred_probabilities.append(probs.item())

        actual_y = actual_y.detach().numpy()
        actual_y = mapIntToClass(int(actual_y[0]),jsonfile)

        # Convert results to dataframe for plotting
        result = pd.DataFrame({'p': pred_probabilities}, index=pred_class_names)
        print(result)
        #plot
        unnormalize = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        plt.figure(figsize=(16, 5))
        ax = plt.subplot(1, 2, 1)
        mid_val = int(len(img)/2)
        plt.interactive(True)
        print(img.shape)
        # ax.set_title(actual_y, size=20)
        # for img_ in img[0].permute(1, 0, 2, 3):
        #     print(img_.shape)
        #     img_ = unnormalize(img_)
        #     img_ = img_.permute(1, 2, 0)
        #     img_ = img_.numpy()
        #     # ax, img = imshow_tensor(img_, ax=ax)
        #     plt.imshow((img_ * 255).astype(np.uint8))
        #     plt.show()

        ax.set_title(actual_y, size=20)
        # Plot a bar plot of predictions
        result.sort_values('p')['p'].plot.barh(color='blue', edgecolor='k', ax=ax)
        plt.xlabel('Predicted Probability')
        plt.tight_layout()
        plt.show()


# %%
"""
We evaluate all of trained model on both the Adult and Kid test split.
To access the checkpoints, kindly send an email to lizfeyisayo@gmail.com
"""

settings = [
    {"model_name":"Adult-SpecificModel", "age":"adults","checkpoint":"model_save\checkpoint_vibrant-waterfall-67_adults.pt","test":"data/Data_Csv/TestSplit-adults.csv"},
    {"model_name":"Adult-SpecificModel", "age":"kids","checkpoint":"model_save\checkpoint_vibrant-waterfall-67_adults.pt","test":"data/Data_Csv/TestSplit-kids.csv"},
    {"model_name":"Kid-SpecificModel", "age":"kids","checkpoint":"model_save\checkpoint_vibrant-dew-70_kids.pt","test":"data/Data_Csv/TestSplit-kids.csv"},
    {"model_name":"Kid-SpecificModel", "age":"adults","checkpoint":"model_save\checkpoint_vibrant-dew-70_kids.pt","test":"data/Data_Csv/TestSplit-adults.csv"},
    {"model_name":"Mixed Model1", "age":"kids","checkpoint":"model_save\checkpoint_dauntless-yogurt-84_kids.pt","test":"data/Data_Csv/TestSplit-kids.csv"},
    {"model_name":"Mixed Model1", "age":"adults","checkpoint":"model_save\checkpoint_dauntless-yogurt-84_kids.pt","test":"data/Data_Csv/TestSplit-adults.csv"},
    {"model_name":"Mixed Model2", "age":"kids","checkpoint":"model_save\checkpoint_genial-deluge-88_kids.pt","test":"data/Data_Csv/TestSplit-kids.csv"},
    {"model_name":"Mixed Model2", "age":"adults","checkpoint":"model_save\checkpoint_genial-deluge-88_kids.pt","test":"data/Data_Csv/TestSplit-adults.csv"}
]

for setting in settings:


    age = setting["age"]
    checkpoint_path = setting['checkpoint']
    model_name = setting['model_name']
    test_csv = setting['test']
#     model_name="Kid-SpecificModel" #if kids model put name as : Kid-SpecificModel

    model = SlowFast(Bottleneck, [3, 4, 6, 3],num_classes=21)
    state_dict = torch.load(checkpoint_path,map_location="cuda")#change checkpoint
    num_ftrs = model.fc.in_features
    model.fc =  nn.Linear(num_ftrs, 21)
    model.load_state_dict(state_dict)
    model.epoch = 0



    # %%


    test = MyDataset(test_csv,mode='test')
    test_dataloader = DataLoader(test,batch_size= 50,shuffle=True)

    file = open("results/Accuracies.txt", "a")



    # %%
    if torch.cuda.is_available():
        model.cuda()

    def plotHeatMap(df_cm):
        heatmap = sns.heatmap(df_cm, annot=True, fmt=".2f")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        heatmap.get_figure().savefig(f"results/{model_name}_{age}.png")
        plt.clf()

    def openJson(age):
        with open(f'test_len_{age}.json') as json_file:
            data = json.load(json_file)
        return data

    def run_statistics(loader, model):
        num_correct = 0
        num_samples = 0
        model.eval()
        accuracy1 = Accuracy(top_k=1)
        accuracy5 = Accuracy(top_k=5)
        nb_classes=21
        confusion_matrix = torch.zeros(nb_classes, nb_classes)
        len_dict = openJson(age)
        idx_to_class =  {val:key for key,val in test.class_to_idx.items()}
        with torch.no_grad():
            for x, y in tqdm(loader):
                x = x.to(device="cuda")
                y = y.to(device="cuda")
                x, y = x.cuda(), y.cuda()
                scores = model(x).cpu()
                _, predictions = scores.max(1)
                y= y.cpu()
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

                #top-k 1 and 5

                accuracy1(scores, y)
                accuracy5(scores, y)

                #confusion_matrix

                for t, p in zip(y.view(-1), predictions.view(-1)):
                    label = idx_to_class[t.item()]
                    len_label = len_dict[label]
                    confusion_matrix[t.long(), p.long()] += 1

            file.write(f'Model name:{model_name}_{age}\n')
            print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
            file.write(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}\n')
            print(f'Top-1 accuracy: {accuracy1.compute()}, Top-5 accuracy {accuracy5.compute()}')
            file.write(f'Top-1 accuracy: {accuracy1.compute()}, Top-5 accuracy {accuracy5.compute()}\n')
            if model_name =="Adult-SpecificModel" and age == "adults":
                idx_to_class =  {val:key for key,val in test.class_to_idx.items()}
                class_names = [idx_to_class[x] for x in range(len(idx_to_class))]
                plt.figure(figsize=(34,38))
                for ind,class_name in enumerate(class_names):
                    n_class = len_dict[class_name]
                    confusion_matrix[ind,:] /= n_class
                confusion_matrix = (confusion_matrix*100)
                df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(float)
                plotHeatMap(df_cm)

                #per class accuracy
                acc_per_class = (confusion_matrix.diag()/confusion_matrix.sum(1))
                acc_per_class = acc_per_class.tolist()
                df_acc_per_class = pd.DataFrame({"Class":class_names,"Accuracy":acc_per_class})
                df_acc_per_class.to_csv(f"results/{model_name}_{age}.csv")

            elif model_name =="Adult-SpecificModel" and age == "kids":
                adult_test = MyDataset(f"data/Data_Csv/TestSplit-adults.csv",mode='test')#change to kids when running kid-specific model
                idx_to_class =  {val:key for key,val in test.class_to_idx.items()}
                idx_to_class_adult =  {val:key for key,val in adult_test.class_to_idx.items()}

                class_names = [idx_to_class[x] for x in range(len(idx_to_class))]
                plt.figure(figsize=(34,38))
                for ind,class_name in enumerate(class_names):
                    n_class = len_dict[class_name]
                    confusion_matrix[ind,:] /= n_class
                confusion_matrix = (confusion_matrix*100)
                df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=sorted(idx_to_class_adult.values())).astype(float)
                plotHeatMap(df_cm)
                #per class accuracy
                acc_per_class = (confusion_matrix.diag()/confusion_matrix.sum(1))
                acc_per_class = acc_per_class.tolist()
                df_acc_per_class = pd.DataFrame({"Class":class_names,"Accuracy":acc_per_class})
                df_acc_per_class.to_csv(f"results/{model_name}_{age}.csv")

            elif model_name =="Kid-SpecificModel" and age == "kids":
                idx_to_class =  {val:key for key,val in test.class_to_idx.items()}
                class_names = [idx_to_class[x] for x in range(len(idx_to_class))]
                plt.figure(figsize=(34,38))
                for ind,class_name in enumerate(class_names):
                    n_class = len_dict[class_name]
                    confusion_matrix[ind,:] /= n_class
                confusion_matrix = (confusion_matrix*100)
                df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(float)
                plotHeatMap(df_cm)

                #per class accuracy
                acc_per_class = (confusion_matrix.diag()/confusion_matrix.sum(1))
                acc_per_class = acc_per_class.tolist()
                df_acc_per_class = pd.DataFrame({"Class":class_names,"Accuracy":acc_per_class})
                df_acc_per_class.to_csv(f"results/{model_name}_{age}.csv")
            elif model_name =="Kid-SpecificModel" and age == "adults":
                kids_test = MyDataset(f"data/Data_Csv/TestSplit-kids.csv",mode='test')
                idx_to_class =  {val:key for key,val in test.class_to_idx.items()}
                idx_to_class_kid =  {val:key for key,val in kids_test.class_to_idx.items()}

                class_names = [idx_to_class[x] for x in range(len(idx_to_class))]
                plt.figure(figsize=(34,38))
                for ind,class_name in enumerate(class_names):
                    n_class = len_dict[class_name]
                    confusion_matrix[ind,:] /= n_class
                confusion_matrix = (confusion_matrix*100)
                df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=sorted(idx_to_class_kid.values())).astype(float)
                plotHeatMap(df_cm)

                #per class accuracy

                acc_per_class = (confusion_matrix.diag()/confusion_matrix.sum(1))
                acc_per_class = acc_per_class.tolist()
                df_acc_per_class = pd.DataFrame({"Class":class_names,"Accuracy":acc_per_class})
                df_acc_per_class.to_csv(f"results/{model_name}_{age}.csv")

            elif (model_name =="Mixed Model1" or model_name =="Mixed Model2") and age=="kids":
                mixed_class = MyDataset(f"data/Data_Csv/train_mixed1.csv",mode='test')
                idx_to_class_mixed =  {val:key for key,val in mixed_class.class_to_idx.items()}
                idx_to_class =  {val:key for key,val in test.class_to_idx.items()}

                class_names = [idx_to_class[x] for x in range(len(idx_to_class))]
                class_names_mixed = [idx_to_class_mixed[x] for x in range(len(idx_to_class_mixed))]
                plt.figure(figsize=(34,38))
                for ind,class_name in enumerate(class_names):
                    n_class = len_dict[class_name]
                    confusion_matrix[ind,:] /= n_class
                confusion_matrix = (confusion_matrix*100)
                df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names_mixed).astype(float)
                plotHeatMap(df_cm)

                #per class accuracy

                acc_per_class = (confusion_matrix.diag()/confusion_matrix.sum(1))
                acc_per_class = acc_per_class.tolist()
                df_acc_per_class = pd.DataFrame({"Class":class_names,"Accuracy":acc_per_class})
                df_acc_per_class.to_csv(f"results/{model_name}_{age}.csv")

            elif (model_name =="Mixed Model1" or model_name =="Mixed Model2") and age=="adults":
                mixed_class = MyDataset(f"data//Data_Csv/train_mixed1.csv",mode='test')
                idx_to_class_mixed =  {val:key for key,val in mixed_class.class_to_idx.items()}
                idx_to_class =  {val:key for key,val in test.class_to_idx.items()}

                class_names = [idx_to_class[x] for x in range(len(idx_to_class))]
                class_names_mixed = [idx_to_class_mixed[x] for x in range(len(idx_to_class_mixed))]
                plt.figure(figsize=(34,38))
                for ind,class_name in enumerate(class_names):
                    n_class = len_dict[class_name]
                    confusion_matrix[ind,:] /= n_class
                confusion_matrix = (confusion_matrix*100)
                df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names_mixed).astype(float)
                plotHeatMap(df_cm)

                #per class accuracy
                acc_per_class = (confusion_matrix.diag()/confusion_matrix.sum(1))
                acc_per_class = acc_per_class.tolist()
                df_acc_per_class = pd.DataFrame({"Class":class_names,"Accuracy":acc_per_class})
                df_acc_per_class.to_csv(f"results/{model_name}_{age}.csv")
        file.close()




# %%
    run_statistics(test_dataloader, model)
