from pathlib import Path
import os
import pickle
import pandas as pd
import numpy as np
import time
import copy

import torch
import torch.utils.data as Data
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt


from sklearn.utils import check_matplotlib_support
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

import cv2
import csv
from PIL import Image

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

# MODEL SETTINGS
args = dict()                 # batch size for training
args['lr'] = 0.01                         # learning rate
args['weight_decay'] = 1e-5
args['batch_size'] = 64
args['epochs'] = 25
args['dropout'] = 0.2
args['output_dim'] = 2                    # number of output classes
args['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args['img_embs_size'] = 128
# MOUNT DRIVE
directory_path = '../capstone'

model_path = os.path.join(directory_path, 'model7_late_fusion_bin_cnn_w_lr_scheduler.pth')

caption_bert_path = os.path.join(directory_path,'Final Data', "caption_1109_bert_128.data")
caption_bert_path_2 = os.path.join(directory_path,'Final Data', "caption_1109_bert_768.data")

score_data_path = os.path.join(directory_path, 'Final Data',"user_scores_normalized.csv")
user_feature_path = os.path.join(directory_path, 'Final Data',"one_hot_user_features_complete.csv")

img_embs_path = os.path.join(directory_path,'Final Data',"img_embs_base_512.data")
img_path = os.path.join(directory_path,'Final Data', "ads_no_category")

# DATA SETTINGS
modality_option = 'both' # 'both', 'single-image', or 'single-caption'
caption_version = 'short' # 'short' or 'long'

# LOAD DATA
if caption_version == 'short':
  file = open(caption_bert_path, 'rb')
else:
  print('long')
  file = open(caption_bert_path_2, 'rb')
captions_bert = pickle.load(file)
captions_bert = captions_bert.data # remove recording of gradient
captions_bert = captions_bert.detach().numpy()

# Reading image files
with open(os.path.join(directory_path,'Final Data',"Augmented-ads-16_cleaned1.csv"), 'r') as f:
    reader = csv.reader(f, delimiter=',')
    headers = next(reader)

ads = [i[3:] for i in headers[6:-11]]

images = []

# Load in the images
print(img_path)

# Reshape image files to same dim (500, 500, 3)
for filepath in ads:
    img = Image.open(img_path+'/'+filepath+'.png')
    new_image = img.convert('RGB')
    new_image = new_image.resize((224, 224))
    new_image = np.asarray(new_image)
    images.append(new_image)

images = np.array(images)
tensor_images = torch.from_numpy(images)
tensor_images = torch.permute(tensor_images, (0,3,1,2)).contiguous()

scores = pd.read_csv(score_data_path).to_numpy()
scores = scores[:, 1:]  # filter out the first column, which is faulty

# convert to binary (scores >=4 --> class 1, scores <4 --> class 0)
scores = scores.reshape(-1, )
filter = (scores >= 0.6).astype(int)
scores = filter.reshape((2000, 300))

# user features
features_df = pd.read_csv(user_feature_path).to_numpy()
features_df = features_df[:, 3:]  # filter out the first three columns, which are faulty

cv_type = ["train", "val", "test"]
feature_type = ["ad", "user"]
for cv in cv_type:
    idx = []
    y = []
    X_image = []
    X_caption = []
    X_user = []
    ad_idx = []
    user_idx = []
    for f in feature_type:
        idx_path = os.path.join(directory_path, 'Final Data', '{}_{}_split.txt'.format(cv, f))
        idx.append(np.loadtxt(idx_path, dtype=int))

    score = scores[idx[1], :][:, idx[0]]
    score = score.reshape(-1, )
    # Score: User #1 - 300 images score; User #2 - 300 images score etc.
    # for i in idx[0]:
    #     X_image.append(images[i])
    for i in idx[1]:
        tmp_x = np.expand_dims(features_df[i, :], axis=1)
        X_user.append(tmp_x)
        user_idx.extend([i] * len(idx[0]))
        for j, val in enumerate(idx[0]):
            # tmp_embed = np.expand_dims(img_embs[j, :], axis=1)
            # X_image.append(np.row_stack((tmp_x, tmp_embed)))
            tmp_embed = np.expand_dims(captions_bert[j, :], axis=1)
            X_caption.append(np.row_stack((tmp_x, tmp_embed)))
            ad_idx.append(val)
            # the 1st 141 dims are user_features, the last 768 dims are captions
    # X_image = np.array(X_image)
    X_caption = np.squeeze(np.array(X_caption))

    # X_image = torch.from_numpy(X_image)
    # X_image = torch.permute(X_image, (0,3,1,2)).contiguous()
    X_caption = torch.from_numpy(X_caption)
    # X_user = torch.from_numpy(X_user)

    if cv == "train":
        # X_image_train = X_image
        X_user_train = X_user
        X_caption_train = X_caption
        y_train = score
        ad_idx_train = ad_idx
        user_idx_train = user_idx
    elif cv == "val":
        # X_image_val = X_image
        X_user_val = X_user
        X_caption_val = X_caption
        y_val = score
        ad_idx_val = ad_idx
        user_idx_val = user_idx
    elif cv == "test":
        # X_image_test = X_image
        X_user_test = X_user
        X_caption_test = X_caption
        y_test = score
        ad_idx_test = ad_idx
        user_idx_test = user_idx

# print("Training dataset: ", X_image_train.shape, y_train.shape)
# print("Validation dataset: ", X_image_val.shape, y_val.shape)
# print("Testing dataset: ", X_image_test.shape, y_test.shape)

print("High ratings in train: ",np.round(np.sum(y_train)/len(y_train)*100,2),"%")
print("Low ratings in train: ", np.round((len(y_train)-np.sum(y_train))/len(y_train)*100,2),"%")

print("High ratings in val: ",np.round(np.sum(y_val)/len(y_val)*100,2),"%")
print("Low ratings in val: ", np.round((len(y_val)-np.sum(y_val))/len(y_val)*100,2),"%")

print("High ratings in test: ",np.round(np.sum(y_test)/len(y_test)*100,2),"%")
print("Low ratings in test: ", np.round((len(y_test)-np.sum(y_test))/len(y_test)*100,2),"%")

ad_idx_val = np.array(ad_idx_val)
ad_idx_test = np.array(ad_idx_test)
ad_idx_train = np.array(ad_idx_train)

class someDataset(Data.Dataset):
    def __init__(self, caption_data, user, img_data, label, ad_index, user_index):
        self.caption_data = caption_data
        self.user = user

        self.label = label
        self.user_index = user_index
        self.ad_index = ad_index
        self.preprocess = models.ResNet50_Weights.DEFAULT.transforms()
        self.img_data = self.preprocess(img_data)
    def __len__(self):
        return len(self.caption_data)
    def __getitem__(self, index): # caption_inputs, image_inputs, labels, ad_indices
        return self.caption_data[index], self.user[self.user_index[index]], self.img_data[self.ad_index[index]]/255., self.label[index]

test_loader = Data.DataLoader(someDataset(X_caption_test, features_df, tensor_images,y_test, ad_idx_train, user_idx_test), shuffle=False, batch_size=args['batch_size'], drop_last=True)

# INITIALIZE MODEL, LOSS FUNCTION, AND OPTIMIZER

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, img_embs_size, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, img_embs_size)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,img_embs_size)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,img_embs_size)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, img_embs_size, kernel_size=(1,1), stride=(1,1))
        model_ft.img_embs_size = img_embs_size
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, img_embs_size)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, img_embs_size)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,img_embs_size)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, args['img_embs_size'], feature_extract, use_pretrained=True)

# Print the model we just instantiated
print(model_ft)

class MLP_baseline(nn.Module):
    def __init__(self, arg, input_dim) -> None:
        super(MLP_baseline, self).__init__()
        hidden_dim = arg['hidden_dim']
        dropout = arg['dropout']
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.BatchNorm1d(num_features=hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.BatchNorm1d(num_features=hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.BatchNorm1d(num_features=hidden_dim[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim[2], hidden_dim[3]),
            nn.BatchNorm1d(num_features=hidden_dim[3]),
            nn.ReLU(),
            nn.Linear(hidden_dim[3], arg['output_dim'])
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class CNN_baseline(nn.Module):
    def __init__(self, arg, dropout=0.2) -> None:
        super(CNN_baseline, self).__init__()
        hidden_dim = arg['hidden_dim']
        dropout = arg['dropout']
        self.conv_layer = model_ft

        # # load up the ResNet50 model
        # self.model = torchvision.models.resnet18(pretrained=True)
        # # append a new classification top to our feature extractor and pop it
        # # on to the current device
        # self.modelOutputFeats = self.model.fc.in_features

        self.mlp = MLP_baseline(arg, 269)

    def forward(self, user_embed, image):
        x = image
        x = self.conv_layer(x)
        # image_embed = self.conv_layer(image)
        # user_image_embs = combine(user_embed, x)
        user_image_embs = torch.cat((user_embed, x), 1)
        # print(user_image_embs.shape)
        image_scores = self.mlp(user_image_embs)
        return image_scores


class late_fusion_dual_mlp(nn.Module):
    def __init__(self, arg, output_class=5, dropout=0.2) -> None:
        super(late_fusion_dual_mlp, self).__init__()
        self.image_cnn_mlp = CNN_baseline(arg)
        # self.image_mlp = MLP_baseline(arg, 653)
        self.caption_mlp = MLP_baseline(arg, 269)

    def forward(self, user_caption_embs, user, image):
        # image_embed = self.image_mlp(image)
        # user_image_embs = combine(user, image_embed)
        image_scores = self.image_cnn_mlp(user, image)
        # image_scores = self.image_mlp(user, image)
        caption_scores = self.caption_mlp(user_caption_embs)
        scores = image_scores + caption_scores
        return scores


def initialize(arg, class_weights, dataloader):
    model = late_fusion_dual_mlp(arg)
    optimizer = optim.Adam(model.parameters(), lr=arg['lr'], weight_decay=arg['weight_decay'])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=arg['lr'], steps_per_epoch=len(dataloader), epochs=arg['epochs'])
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # WCE
    return model, optimizer, scheduler, criterion

class_weights=compute_class_weight(class_weight ='balanced',classes =np.unique(y_train),y = y_train)
class_weights=torch.tensor(class_weights,dtype=torch.float)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class_weights = class_weights.to(device)
model, optimizer, scheduler, criterion = initialize(args, class_weights, test_loader)
model.to(device)

correct = 0
total = 0
y_pred = []
y_true = []
model.load_state_dict(torch.load(model_path))
model.eval()
with torch.no_grad():
    for i, batch in enumerate(test_loader, 0):
        caption_inputs, user, image_inputs, labels = batch[0].float().to(device), batch[1].float().to(device), batch[2].float().to(device), batch[3].type(torch.LongTensor).to(device)
        # load image missing here (to d0)
        outputs = model(caption_inputs, user, image_inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.append(predicted.cpu().numpy())
        y_true.append(labels.cpu().numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
test_acc = 100 * correct / total


y_true= [y_true[i][0] for i in range(len(y_pred))]
y_pred= [y_pred[i][0] for i in range(len(y_pred))]

acc = accuracy_score(y_true, y_pred)
f1_score = f1_score(y_true, y_pred, average='weighted')

print("Test accuracy:{:.5f}".format(acc))
print("F1 score:{:.5f}".format(f1_score))

# constant for classes
classes = ('Dislike (<4)','Like (>=4)')

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                    columns = [i for i in classes])
ax= plt.subplot()
sn.heatmap(df_cm, annot=True)
# plt.figure(figsize = (12,7))
# plt.title("Model 4: Early Fusion Confusion Matrix")
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('model7_late_fusion_bin_cnn_1 Confusion Matrix');
ax.xaxis.set_ticklabels([i for i in classes]); ax.yaxis.set_ticklabels([i for i in classes]);
# plt.ylabel("True score")
plt.savefig('model7_late_fusion_bin_cnn_1.png')
