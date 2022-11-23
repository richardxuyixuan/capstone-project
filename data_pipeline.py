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


class someDataset(Data.Dataset):
    def __init__(self, caption_data, user_feats, img_data, label, ad_index, user_index):
        self.caption_data = caption_data
        self.user_feats = user_feats
        self.label = label - 1
        self.user_index = user_index
        self.ad_index = ad_index
        self.preprocess = models.ResNet50_Weights.DEFAULT.transforms()
        self.img_data = self.preprocess(img_data)

    def __len__(self):
        return len(self.caption_data)

    def __getitem__(self, index):
        return self.caption_data[index], self.user_feats[self.user_index[index]], self.img_data[self.ad_index[index]], \
               self.label[index]

def get_data(caption_version, args):
    caption_bert_path = os.path.join('Final Data', "caption_1109_bert_128.data")
    caption_bert_path_2 = os.path.join('Final Data', "caption_1109_bert_768.data")

    score_data_path = os.path.join('Final Data', "user_scores_normalized.csv")
    user_feature_path = os.path.join('Final Data', "one_hot_user_features_complete.csv")

    img_embs_path = os.path.join('Final Data', "img_embs_base_512.data")
    img_path = os.path.join('Final Data', "ads_no_category")

    # LOAD DATA
    if caption_version == 'short':
        file = open(caption_bert_path, 'rb')
    else:
        print('long')
        file = open(caption_bert_path_2, 'rb')
    captions_bert = pickle.load(file)
    captions_bert = captions_bert.data  # remove recording of gradient
    captions_bert = captions_bert.detach().numpy()

    # Reading image files
    with open(os.path.join('Final Data', "Augmented-ads-16_cleaned1.csv"), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)

    ads = [i[3:] for i in headers[6:-11]]

    images = []

    # Load in the images
    print(img_path)

    # Reshape image files to same dim (500, 500, 3)
    for filepath in ads:
        img = Image.open(img_path + '/' + filepath + '.png')
        new_image = img.convert('RGB')
        new_image = new_image.resize((224, 224))
        new_image = np.asarray(new_image)
        images.append(new_image)

    images = np.array(images)
    tensor_images = torch.from_numpy(images)
    tensor_images = torch.permute(tensor_images, (0, 3, 1, 2)).contiguous()

    scores = pd.read_csv(score_data_path).to_numpy()
    scores = scores[:, 1:]  # filter out the first column, which is faulty
    scores *= 5

    # user features
    features_df = pd.read_csv(user_feature_path).to_numpy()
    features_df = features_df[:, 3:]  # filter out the first three columns, which are faulty

    cv_type = ["train", "val", "test"]
    feature_type = ["ad", "user"]
    for cv in cv_type:
        idx = []
        X_caption = []
        X_user = []
        ad_idx = []
        user_idx = []
        for f in feature_type:
            idx_path = os.path.join('Final Data', '{}_{}_split.txt'.format(cv, f))
            idx.append(np.loadtxt(idx_path, dtype=int))

        score = scores[idx[1], :][:, idx[0]]
        score = score.reshape(-1, )
        # Score: User #1 - 300 images score; User #2 - 300 images score etc.
        for i in idx[1]:
            tmp_x = np.expand_dims(features_df[i, :], axis=1)
            X_user.append(tmp_x)
            user_idx.extend([i] * len(idx[0]))
            for j in idx[0]:
                tmp_embed = np.expand_dims(captions_bert[j, :], axis=1)
                X_caption.append(np.row_stack((tmp_x, tmp_embed)))
                ad_idx.append(j)
                # the 1st 141 dims are user_features, the last 768 dims are captions
        X_caption = np.squeeze(np.array(X_caption))
        X_caption = torch.from_numpy(X_caption)
        X_user = np.array(X_user)
        if cv == "train":
            X_user_train = X_user
            X_caption_train = X_caption
            y_train = score
            ad_idx_train = ad_idx
            user_idx_train = user_idx
        elif cv == "val":
            X_user_val = X_user
            X_caption_val = X_caption
            y_val = score
            ad_idx_val = ad_idx
            user_idx_val = user_idx
        elif cv == "test":
            X_user_test = X_user
            X_caption_test = X_caption
            y_test = score
            ad_idx_test = ad_idx
            user_idx_test = user_idx

    ad_idx_val = np.array(ad_idx_val)
    ad_idx_test = np.array(ad_idx_test)
    ad_idx_train = np.array(ad_idx_train)


    train_loader = Data.DataLoader(
        someDataset(X_caption_train, features_df, tensor_images, y_train, ad_idx_train, user_idx_train), shuffle=True,
        batch_size=args['batch_size'], drop_last=True)
    val_loader = Data.DataLoader(
        someDataset(X_caption_val, features_df, tensor_images, y_val, ad_idx_val, user_idx_val), shuffle=False,
        batch_size=args['batch_size'], drop_last=True)
    test_loader = Data.DataLoader(
        someDataset(X_caption_test, features_df, tensor_images, y_test, ad_idx_test, user_idx_test), shuffle=False,
        batch_size=args['batch_size'], drop_last=True)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    return train_loader, val_loader, test_loader, class_weights