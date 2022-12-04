import torch
from torch import optim
import torch.nn as nn
from torchvision import datasets, models, transforms
from models import model_image_only, model_caption_only, model_early_fusion, model_late_fusion, model_early_fusion_se, model_baseline, model_baseline_user_bert

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_resnet(model_version, img_embs_size, feature_extract=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    set_parameter_requires_grad(model_ft, feature_extract)
    if 'early_fusion' in model_version:
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
    elif arg['model_version'] == 'early_fusion_se':
        model =model_early_fusion_se(arg, resnet_model)
    elif arg['model_version'] == 'late_fusion':
        model = model_late_fusion(arg, resnet_model)
    elif arg['model_version'] == 'baseline':
        model = model_baseline(arg)
    elif arg['model_version'] == 'baseline_user_bert':
        model = model_baseline_user_bert(arg)

    if arg['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=arg['lr'], weight_decay=arg['weight_decay'])
    elif arg['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr= arg['lr'], momentum=0.9)
    if 'use_focal_loss' in arg.keys():
        if arg['use_focal_loss'] == True:
            criterion = torch.hub.load(
                'adeelh/pytorch-multi-class-focal-loss',
                model='FocalLoss',
                alpha=class_weights,
                gamma=2,
                reduction='mean',
                force_reload=False
            )
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)  # WCE
    elif 'use_bce' in arg.keys():
        if arg['use_bce'] == True:
            criterion = nn.BCEWithLogitsLoss() # BCE
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)  # WCE

    if 'no_class_weights' in arg.keys():
        if arg['no_class_weights']:
            criterion = nn.CrossEntropyLoss()

    # criterion = nn.BCELoss()
    if arg['use_scheduler']:
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=arg['lr'], steps_per_epoch=len(dataloader), epochs=arg['epochs'])
    else:
        scheduler = None
    return model, optimizer, scheduler, criterion