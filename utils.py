import torch
from torch import optim
import torch.nn as nn
from torchvision import datasets, models, transforms
from models import model_image_only, model_caption_only, model_early_fusion, model_late_fusion

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_resnet(model_version, img_embs_size, feature_extract=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    set_parameter_requires_grad(model_ft, feature_extract)
    if model_version == 'early_fusion':
        model_ft = torch.nn.Sequential(*(list(model_ft.children())[:-2]))
    else:
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, img_embs_size)
    return model_ft

def initialize_model(arg, class_weights, dataloader, resnet_model):
    if arg['model_version'] == 'image_only':
        model = model_image_only(arg, resnet_model)
    elif arg['model_version'] == 'caption_only':
        model = model_caption_only(arg, resnet_model)
    elif arg['model_version'] == 'early_fusion':
        model = model_early_fusion(arg, resnet_model)
    elif arg['model_version'] == 'late_fusion':
        model = model_late_fusion(arg, resnet_model)
    if arg['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=arg['lr'], weight_decay=arg['weight_decay'])
    elif arg['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr= arg['lr'], momentum=0.9)
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # WCE
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=arg['lr'], steps_per_epoch=len(dataloader), epochs=arg['epochs'])
    return model, optimizer, scheduler, criterion