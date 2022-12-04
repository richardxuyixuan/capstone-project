import os
import numpy as np
import torch
from data_pipeline import get_new_data, get_data
from utils import initialize_resnet, initialize_model

# MODEL SETTINGS
args = dict()                 # batch size for training
args['caption_version'] = 'image'  # 'short', 'long', 'old', 'image', 'image_tuned'
args['model_version'] = 'caption_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args['checkpoint_name'] = 'image_only_clip_lr_1e-2_bs_64.pth'
args['data'] = 'augmented' # choose between 'augmented' and 'non_augmented'
args['lr'] = 0.01                         # learning rate
args['weight_decay'] = 1e-5
args['batch_size'] = 64
args['epochs'] = 25
args['dropout'] = 0.2
args['output_dim'] = 2                    # number of output classes
args['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args['img_embs_size'] = 128
args['optimizer'] = 'Adam' # 'Adam or SGD'
args['use_scheduler'] = True

args1 = dict()                 # batch size for training
args1['caption_version'] = 'image_tuned'  # 'short', 'long', 'old', 'image', 'image_tuned'
args1['model_version'] = 'caption_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args1['checkpoint_name'] = 'image_only_clip_tuned_lr_1e-2_bs_64.pth'
args1['data'] = 'augmented' # choose between 'augmented' and 'non_augmented'
args1['lr'] = 0.01                         # learning rate
args1['weight_decay'] = 1e-5
args1['batch_size'] = 64
args1['epochs'] = 25
args1['dropout'] = 0.2
args1['output_dim'] = 2                    # number of output classes
args1['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args1['img_embs_size'] = 128
args1['optimizer'] = 'Adam' # 'Adam or SGD'
args1['use_scheduler'] = True

args2 = dict()                 # batch size for training
args2['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args2['model_version'] = 'image_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args2['checkpoint_name'] = 'image_only_resnet50_lr_1e-2_bs_64.pth'
args2['data'] = 'augmented' # choose between 'augmented' and 'non_augmented'
args2['lr'] = 0.01                         # learning rate
args2['weight_decay'] = 1e-5
args2['batch_size'] = 64
args2['epochs'] = 25
args2['dropout'] = 0.2
args2['output_dim'] = 2                    # number of output classes
args2['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args2['img_embs_size'] = 128
args2['optimizer'] = 'Adam' # 'Adam or SGD'
args2['use_scheduler'] = True

args3 = dict()                 # batch size for training
args3['caption_version'] = 'old'  # 'short', 'long', 'old', 'image', 'image_tuned'
args3['model_version'] = 'caption_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args3['checkpoint_name'] = 'caption_only_not_segmented_image_lr_1e-2_bs_64.pth'
args3['data'] = 'augmented' # choose between 'augmented' and 'non_augmented'
args3['lr'] = 0.01                         # learning rate
args3['weight_decay'] = 1e-5
args3['batch_size'] = 64
args3['epochs'] = 25
args3['dropout'] = 0.2
args3['output_dim'] = 2                    # number of output classes
args3['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args3['img_embs_size'] = 128
args3['optimizer'] = 'Adam' # 'Adam or SGD'
args3['use_scheduler'] = True

args4 = dict()                 # batch size for training
args4['caption_version'] = 'long'  # 'short', 'long', 'old', 'image', 'image_tuned'
args4['model_version'] = 'caption_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args4['checkpoint_name'] = 'caption_only_segmented_image_lr_1e-2_bs_64.pth'
args4['data'] = 'augmented' # choose between 'augmented' and 'non_augmented'
args4['lr'] = 0.01                         # learning rate
args4['weight_decay'] = 1e-5
args4['batch_size'] = 64
args4['epochs'] = 25
args4['dropout'] = 0.2
args4['output_dim'] = 2                    # number of output classes
args4['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args4['img_embs_size'] = 128
args4['optimizer'] = 'Adam' # 'Adam or SGD'
args4['use_scheduler'] = True

args5 = dict()                 # batch size for training
args5['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args5['model_version'] = 'caption_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args5['checkpoint_name'] = 'caption_only_segmented_image_pca_lr_1e-2_bs_64.pth'
args5['data'] = 'augmented' # choose between 'augmented' and 'non_augmented'
args5['lr'] = 0.01                         # learning rate
args5['weight_decay'] = 1e-5
args5['batch_size'] = 64
args5['epochs'] = 25
args5['dropout'] = 0.2
args5['output_dim'] = 2                    # number of output classes
args5['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args5['img_embs_size'] = 128
args5['optimizer'] = 'Adam' # 'Adam or SGD'
args5['use_scheduler'] = True

args6 = dict()                 # batch size for training
args6['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args6['model_version'] = 'early_fusion' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args6['checkpoint_name'] = 'early_fusion_lr_1e-2_bs_64.pth'
args6['data'] = 'augmented' # choose between 'augmented' and 'non_augmented'
args6['lr'] = 0.01                         # learning rate
args6['weight_decay'] = 1e-5
args6['batch_size'] = 64
args6['epochs'] = 25
args6['dropout'] = 0.2
args6['output_dim'] = 2                    # number of output classes
args6['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args6['img_embs_size'] = 128
args6['optimizer'] = 'Adam' # 'Adam or SGD'
args6['use_scheduler'] = True

args7 = dict()                 # batch size for training
args7['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args7['model_version'] = 'late_fusion' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args7['checkpoint_name'] = 'late_fusion_lr_1e-2_bs_64.pth'
args7['data'] = 'augmented' # choose between 'augmented' and 'non_augmented'
args7['lr'] = 0.01                         # learning rate
args7['weight_decay'] = 1e-5
args7['batch_size'] = 64
args7['epochs'] = 25
args7['dropout'] = 0.2
args7['output_dim'] = 2                    # number of output classes
args7['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args7['img_embs_size'] = 128
args7['optimizer'] = 'Adam' # 'Adam or SGD'
args7['use_scheduler'] = True

args8 = dict()                 # batch size for training
args8['caption_version'] = 'long'  # 'short', 'long', 'old', 'image', 'image_tuned'
args8['model_version'] = 'early_fusion' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args8['checkpoint_name'] = 'early_fusion_long_caption_lr_1e-2_bs_64.pth'
args8['data'] = 'augmented' # choose between 'augmented' and 'non_augmented'
args8['lr'] = 0.01                         # learning rate
args8['weight_decay'] = 1e-5
args8['batch_size'] = 64
args8['epochs'] = 25
args8['dropout'] = 0.2
args8['output_dim'] = 2                    # number of output classes
args8['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args8['img_embs_size'] = 128
args8['optimizer'] = 'Adam' # 'Adam or SGD'
args8['use_scheduler'] = True

args8_a = dict()                 # batch size for training
args8_a['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args8_a['model_version'] = 'early_fusion_se' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args8_a['checkpoint_name'] = 'early_fusion_se_lr_1e-2_bs_64.pth'
args8_a['data'] = 'augmented' # choose between 'augmented' and 'non_augmented'
args8_a['lr'] = 0.01                         # learning rate
args8_a['weight_decay'] = 1e-5
args8_a['batch_size'] = 64
args8_a['epochs'] = 25
args8_a['dropout'] = 0.2
args8_a['output_dim'] = 2                    # number of output classes
args8_a['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args8_a['img_embs_size'] = 128
args8_a['optimizer'] = 'Adam' # 'Adam or SGD'
args8_a['use_scheduler'] = True

args9 = dict()                 # batch size for training
args9['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args9['model_version'] = 'early_fusion' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args9['checkpoint_name'] = 'early_fusion_no_scheduler_lr_1e-3_bs_64.pth'
args9['data'] = 'augmented' # choose between 'augmented' and 'non_augmented'
args9['lr'] = 0.001                         # learning rate
args9['weight_decay'] = 1e-5
args9['batch_size'] = 64
args9['epochs'] = 25
args9['dropout'] = 0.2
args9['output_dim'] = 2                    # number of output classes
args9['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args9['img_embs_size'] = 128
args9['optimizer'] = 'Adam' # 'Adam or SGD'
args9['use_scheduler'] = False

argsa = dict()                 # batch size for training
argsa['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
argsa['model_version'] = 'early_fusion' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
argsa['checkpoint_name'] = 'early_fusion_focal_loss_lr_1e-2_bs_64.pth'
argsa['data'] = 'augmented' # choose between 'augmented' and 'non_augmented'
argsa['lr'] = 0.01                         # learning rate
argsa['weight_decay'] = 1e-5
argsa['batch_size'] = 64
argsa['epochs'] = 25
argsa['dropout'] = 0.2
argsa['output_dim'] = 2                    # number of output classes
argsa['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
argsa['img_embs_size'] = 128
argsa['optimizer'] = 'Adam' # 'Adam or SGD'
argsa['use_scheduler'] = True
argsa['use_focal_loss'] = True

argsb = dict()                 # batch size for training
argsb['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
argsb['model_version'] = 'early_fusion' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
argsb['checkpoint_name'] = 'early_fusion_non_augmented_data_lr_1e-2_bs_64.pth'
argsb['data'] = 'non_augmented' # choose between 'augmented' and 'non_augmented'
argsb['lr'] = 0.001                         # learning rate
argsb['weight_decay'] = 1e-5
argsb['batch_size'] = 64
argsb['epochs'] = 25
argsb['dropout'] = 0.2
argsb['output_dim'] = 2                    # number of output classes
argsb['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
argsb['img_embs_size'] = 128
argsb['optimizer'] = 'Adam' # 'Adam or SGD'
argsb['use_scheduler'] = False

argsn = dict()                 # batch size for training
argsn['caption_version'] = 'image'  # 'short', 'long', 'old', 'image', 'image_tuned'
argsn['model_version'] = 'caption_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
argsn['checkpoint_name'] = 'image_only_non_augmented_clip_lr_1e-2_bs_64.pth'
argsn['data'] = 'non_augmented' # choose between 'augmented' and 'non_augmented'
argsn['lr'] = 0.001                         # learning rate
argsn['weight_decay'] = 1e-5
argsn['batch_size'] = 64
argsn['epochs'] = 25
argsn['dropout'] = 0.2
argsn['output_dim'] = 2                    # number of output classes
argsn['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
argsn['img_embs_size'] = 128
argsn['optimizer'] = 'Adam' # 'Adam or SGD'
argsn['use_scheduler'] = False

args1n = dict()                 # batch size for training
args1n['caption_version'] = 'image_tuned'  # 'short', 'long', 'old', 'image', 'image_tuned'
args1n['model_version'] = 'caption_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args1n['checkpoint_name'] = 'image_only_non_augmented_clip_tuned_lr_1e-2_bs_64.pth'
args1n['data'] = 'non_augmented' # choose between 'augmented' and 'non_augmented'
args1n['lr'] = 0.001                         # learning rate
args1n['weight_decay'] = 1e-5
args1n['batch_size'] = 64
args1n['epochs'] = 25
args1n['dropout'] = 0.2
args1n['output_dim'] = 2                    # number of output classes
args1n['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args1n['img_embs_size'] = 128
args1n['optimizer'] = 'Adam' # 'Adam or SGD'
args1n['use_scheduler'] = False

args2n = dict()                 # batch size for training
args2n['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args2n['model_version'] = 'image_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args2n['checkpoint_name'] = 'image_only_non_augmented_resnet50_lr_1e-2_bs_64.pth'
args2n['data'] = 'non_augmented' # choose between 'augmented' and 'non_augmented'
args2n['lr'] = 0.001                         # learning rate
args2n['weight_decay'] = 1e-5
args2n['batch_size'] = 64
args2n['epochs'] = 25
args2n['dropout'] = 0.2
args2n['output_dim'] = 2                    # number of output classes
args2n['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args2n['img_embs_size'] = 128
args2n['optimizer'] = 'Adam' # 'Adam or SGD'
args2n['use_scheduler'] = False

args3n = dict()                 # batch size for training
args3n['caption_version'] = 'old'  # 'short', 'long', 'old', 'image', 'image_tuned'
args3n['model_version'] = 'caption_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args3n['checkpoint_name'] = 'caption_only_non_augmented_not_segmented_image_lr_1e-2_bs_64.pth'
args3n['data'] = 'non_augmented' # choose between 'augmented' and 'non_augmented'
args3n['lr'] = 0.001                         # learning rate
args3n['weight_decay'] = 1e-5
args3n['batch_size'] = 64
args3n['epochs'] = 25
args3n['dropout'] = 0.2
args3n['output_dim'] = 2                    # number of output classes
args3n['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args3n['img_embs_size'] = 128
args3n['optimizer'] = 'Adam' # 'Adam or SGD'
args3n['use_scheduler'] = False

args4n = dict()                 # batch size for training
args4n['caption_version'] = 'long'  # 'short', 'long', 'old', 'image', 'image_tuned'
args4n['model_version'] = 'caption_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args4n['checkpoint_name'] = 'caption_only_non_augmented_segmented_image_lr_1e-2_bs_64.pth'
args4n['data'] = 'non_augmented' # choose between 'augmented' and 'non_augmented'
args4n['lr'] = 0.001                         # learning rate
args4n['weight_decay'] = 1e-5
args4n['batch_size'] = 64
args4n['epochs'] = 25
args4n['dropout'] = 0.2
args4n['output_dim'] = 2                    # number of output classes
args4n['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args4n['img_embs_size'] = 128
args4n['optimizer'] = 'Adam' # 'Adam or SGD'
args4n['use_scheduler'] = False

args5n = dict()                 # batch size for training
args5n['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args5n['model_version'] = 'caption_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args5n['checkpoint_name'] = 'caption_only_non_augmented_segmented_image_pca_lr_1e-2_bs_64.pth'
args5n['data'] = 'non_augmented' # choose between 'augmented' and 'non_augmented'
args5n['lr'] = 0.001                         # learning rate
args5n['weight_decay'] = 1e-5
args5n['batch_size'] = 64
args5n['epochs'] = 25
args5n['dropout'] = 0.2
args5n['output_dim'] = 2                    # number of output classes
args5n['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args5n['img_embs_size'] = 128
args5n['optimizer'] = 'Adam' # 'Adam or SGD'
args5n['use_scheduler'] = False

args7n = dict()                 # batch size for training
args7n['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args7n['model_version'] = 'late_fusion' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args7n['checkpoint_name'] = 'late_fusion_non_augmented_lr_1e-2_bs_64.pth'
args7n['data'] = 'non_augmented' # choose between 'augmented' and 'non_augmented'
args7n['lr'] = 0.001                         # learning rate
args7n['weight_decay'] = 1e-5
args7n['batch_size'] = 64
args7n['epochs'] = 25
args7n['dropout'] = 0.2
args7n['output_dim'] = 2                    # number of output classes
args7n['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args7n['img_embs_size'] = 128
args7n['optimizer'] = 'Adam' # 'Adam or SGD'
args7n['use_scheduler'] = False

args6b = dict()                 # batch size for training
args6b['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args6b['model_version'] = 'early_fusion' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args6b['checkpoint_name'] = 'early_fusion_lr_1e-2_bs_64.pth'
args6b['data'] = 'augmented' # choose between 'augmented' and 'non_augmented'
args6b['lr'] = 0.01                         # learning rate
args6b['weight_decay'] = 1e-5
args6b['batch_size'] = 64
args6b['epochs'] = 25
args6b['dropout'] = 0.2
args6b['output_dim'] = 1                    # number of output classes
args6b['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args6b['img_embs_size'] = 128
args6b['optimizer'] = 'Adam' # 'Adam or SGD'
args6b['use_scheduler'] = True
args6b['use_bce'] = True

args6c = dict()                 # batch size for training
args6c['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args6c['model_version'] = 'early_fusion' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args6c['checkpoint_name'] = 'early_fusion_bce_non_aug_no_scheduler_lr_1e-2_bs_64.pth'
args6c['data'] = 'non_augmented' # choose between 'augmented' and 'non_augmented'
args6c['lr'] = 0.01                         # learning rate
args6c['weight_decay'] = 1e-5
args6c['batch_size'] = 64
args6c['epochs'] = 25
args6c['dropout'] = 0.2
args6c['output_dim'] = 1                    # number of output classes
args6c['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args6c['img_embs_size'] = 128
args6c['optimizer'] = 'Adam' # 'Adam or SGD'
args6c['use_scheduler'] = False
args6c['use_bce'] = True

args6d = dict()                 # batch size for training
args6d['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args6d['model_version'] = 'early_fusion' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args6d['checkpoint_name'] = 'early_fusion_focal_non_aug_no_scheduler_lr_1e-2_bs_64.pth'
args6d['data'] = 'non_augmented' # choose between 'augmented' and 'non_augmented'
args6d['lr'] = 0.01                         # learning rate
args6d['weight_decay'] = 1e-5
args6d['batch_size'] = 64
args6d['epochs'] = 25
args6d['dropout'] = 0.2
args6d['output_dim'] = 2                    # number of output classes
args6d['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args6d['img_embs_size'] = 128
args6d['optimizer'] = 'Adam' # 'Adam or SGD'
args6d['use_scheduler'] = False
args6d['use_focal_loss'] = True

argse = dict()                 # batch size for training
argse['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
argse['model_version'] = 'caption_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
argse['checkpoint_name'] = 'caption_only_bce_non_aug_no_scheduler_lr_1e-2_bs_64.pth'
argse['data'] = 'non_augmented' # choose between 'augmented' and 'non_augmented'
argse['lr'] = 0.01                         # learning rate
argse['weight_decay'] = 1e-5
argse['batch_size'] = 64
argse['epochs'] = 25
argse['dropout'] = 0.2
argse['output_dim'] = 1                    # number of output classes
argse['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
argse['img_embs_size'] = 128
argse['optimizer'] = 'Adam' # 'Adam or SGD'
argse['use_scheduler'] = False
argse['use_bce'] = True

argsf = dict()                 # batch size for training
argsf['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
argsf['model_version'] = 'caption_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
argsf['checkpoint_name'] = 'caption_only_focal_non_aug_no_scheduler_lr_1e-2_bs_64.pth'
argsf['data'] = 'non_augmented' # choose between 'augmented' and 'non_augmented'
argsf['lr'] = 0.01                         # learning rate
argsf['weight_decay'] = 1e-5
argsf['batch_size'] = 64
argsf['epochs'] = 25
argsf['dropout'] = 0.2
argsf['output_dim'] = 2                    # number of output classes
argsf['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
argsf['img_embs_size'] = 128
argsf['optimizer'] = 'Adam' # 'Adam or SGD'
argsf['use_scheduler'] = False
argsf['use_focal_loss'] = True

argsg = dict()                 # batch size for training
argsg['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
argsg['model_version'] = 'caption_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
argsg['checkpoint_name'] = 'caption_only_augmented_no_scheduler_lr_1e-2_bs_64.pth'
argsg['data'] = 'augmented' # choose between 'augmented' and 'non_augmented'
argsg['lr'] = 0.01                         # learning rate
argsg['weight_decay'] = 1e-5
argsg['batch_size'] = 64
argsg['epochs'] = 25
argsg['dropout'] = 0.2
argsg['output_dim'] = 2                    # number of output classes
argsg['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
argsg['img_embs_size'] = 128
argsg['optimizer'] = 'Adam' # 'Adam or SGD'
argsg['use_scheduler'] = False

argsh = dict()                 # batch size for training
argsh['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
argsh['model_version'] = 'image_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
argsh['checkpoint_name'] = 'image_only_bce_non_aug_no_scheduler_lr_1e-2_bs_64.pth'
argsh['data'] = 'non_augmented' # choose between 'augmented' and 'non_augmented'
argsh['lr'] = 0.01                         # learning rate
argsh['weight_decay'] = 1e-5
argsh['batch_size'] = 64
argsh['epochs'] = 25
argsh['dropout'] = 0.2
argsh['output_dim'] = 1                    # number of output classes
argsh['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
argsh['img_embs_size'] = 128
argsh['optimizer'] = 'Adam' # 'Adam or SGD'
argsh['use_scheduler'] = False
argsh['use_bce'] = True

argsi = dict()                 # batch size for training
argsi['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
argsi['model_version'] = 'image_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
argsi['checkpoint_name'] = 'image_only_focal_non_aug_no_scheduler_lr_1e-2_bs_64.pth'
argsi['data'] = 'non_augmented' # choose between 'augmented' and 'non_augmented'
argsi['lr'] = 0.01                         # learning rate
argsi['weight_decay'] = 1e-5
argsi['batch_size'] = 64
argsi['epochs'] = 25
argsi['dropout'] = 0.2
argsi['output_dim'] = 2                    # number of output classes
argsi['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
argsi['img_embs_size'] = 128
argsi['optimizer'] = 'Adam' # 'Adam or SGD'
argsi['use_scheduler'] = False
argsi['use_focal_loss'] = True

argsj = dict()                 # batch size for training
argsj['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
argsj['model_version'] = 'image_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
argsj['checkpoint_name'] = 'image_only_augmented_no_scheduler_lr_1e-2_bs_64.pth'
argsj['data'] = 'augmented' # choose between 'augmented' and 'non_augmented'
argsj['lr'] = 0.01                         # learning rate
argsj['weight_decay'] = 1e-5
argsj['batch_size'] = 64
argsj['epochs'] = 25
argsj['dropout'] = 0.2
argsj['output_dim'] = 2                    # number of output classes
argsj['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
argsj['img_embs_size'] = 128
argsj['optimizer'] = 'Adam' # 'Adam or SGD'
argsj['use_scheduler'] = False

args_base = dict()                 # batch size for training
args_base['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args_base['model_version'] = 'baseline' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args_base['checkpoint_name'] = 'baseline_bce_non_augmented_no_scheduler_lr_1e-2_bs_64.pth'
args_base['data'] = 'non_augmented' # choose between 'augmented' and 'non_augmented'
args_base['lr'] = 0.01                         # learning rate
args_base['weight_decay'] = 1e-5
args_base['batch_size'] = 64
args_base['epochs'] = 25
args_base['dropout'] = 0.2
args_base['output_dim'] = 1                    # number of output classes
args_base['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args_base['img_embs_size'] = 128
args_base['optimizer'] = 'Adam' # 'Adam or SGD'
args_base['use_scheduler'] = False
args_base['use_bce'] = True

args_base1 = dict()                 # batch size for training
args_base1['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args_base1['model_version'] = 'baseline' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args_base1['checkpoint_name'] = 'baseline_wce_non_augmented_no_scheduler_lr_1e-2_bs_64.pth'
args_base1['data'] = 'non_augmented' # choose between 'augmented' and 'non_augmented'
args_base1['lr'] = 0.01                         # learning rate
args_base1['weight_decay'] = 1e-5
args_base1['batch_size'] = 64
args_base1['epochs'] = 25
args_base1['dropout'] = 0.2
args_base1['output_dim'] = 2                   # number of output classes
args_base1['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args_base1['img_embs_size'] = 128
args_base1['optimizer'] = 'Adam' # 'Adam or SGD'
args_base1['use_scheduler'] = False

args_base2 = dict()                 # batch size for training
args_base2['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args_base2['model_version'] = 'baseline' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args_base2['checkpoint_name'] = 'baseline_focal_non_augmented_no_scheduler_lr_1e-2_bs_64.pth'
args_base2['data'] = 'non_augmented' # choose between 'augmented' and 'non_augmented'
args_base2['lr'] = 0.01                         # learning rate
args_base2['weight_decay'] = 1e-5
args_base2['batch_size'] = 64
args_base2['epochs'] = 25
args_base2['dropout'] = 0.2
args_base2['output_dim'] = 2                    # number of output classes
args_base2['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args_base2['img_embs_size'] = 128
args_base2['optimizer'] = 'Adam' # 'Adam or SGD'
args_base2['use_scheduler'] = False
args_base2['use_focal_loss'] = True

args_base3 = dict()                 # batch size for training
args_base3['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args_base3['model_version'] = 'baseline' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args_base3['checkpoint_name'] = 'baseline_bce_no_scheduler_lr_1e-2_bs_64.pth'
args_base3['data'] = 'augmented' # choose between 'augmented' and 'non_augmented'
args_base3['lr'] = 0.01                         # learning rate
args_base3['weight_decay'] = 1e-5
args_base3['batch_size'] = 64
args_base3['epochs'] = 25
args_base3['dropout'] = 0.2
args_base3['output_dim'] = 1                    # number of output classes
args_base3['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args_base3['img_embs_size'] = 128
args_base3['optimizer'] = 'Adam' # 'Adam or SGD'
args_base3['use_scheduler'] = False
args_base3['use_bce'] = True

args_base4 = dict()                 # batch size for training
args_base4['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args_base4['model_version'] = 'baseline' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args_base4['checkpoint_name'] = 'baseline_bce_lr_1e-2_bs_64.pth'
args_base4['data'] = 'augmented' # choose between 'augmented' and 'non_augmented'
args_base4['lr'] = 0.01                         # learning rate
args_base4['weight_decay'] = 1e-5
args_base4['batch_size'] = 64
args_base4['epochs'] = 25
args_base4['dropout'] = 0.2
args_base4['output_dim'] = 1                    # number of output classes
args_base4['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args_base4['img_embs_size'] = 128
args_base4['optimizer'] = 'Adam' # 'Adam or SGD'
args_base4['use_scheduler'] = True
args_base4['use_bce'] = True

args_base5 = dict()                 # batch size for training
args_base5['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args_base5['model_version'] = 'baseline_user_bert' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args_base5['checkpoint_name'] = 'baseline_user_bert_bce_non_augmented_no_scheduler_lr_1e-2_bs_64.pth'
args_base5['data'] = 'non_augmented' # choose between 'augmented' and 'non_augmented'
args_base5['lr'] = 0.01                         # learning rate
args_base5['weight_decay'] = 1e-5
args_base5['batch_size'] = 64
args_base5['epochs'] = 25
args_base5['dropout'] = 0.2
args_base5['output_dim'] = 1                    # number of output classes
args_base5['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args_base5['img_embs_size'] = 128
args_base5['optimizer'] = 'Adam' # 'Adam or SGD'
args_base5['use_scheduler'] = False
args_base5['use_bce'] = True
args_base5['user_bert'] = True

args_base6 = dict()                 # batch size for training
args_base6['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args_base6['model_version'] = 'baseline' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args_base6['checkpoint_name'] = 'baseline_5_class_wce_non_augmented_no_scheduler_lr_1e-2_bs_64.pth'
args_base6['data'] = 'non_augmented' # choose between 'augmented' and 'non_augmented'
args_base6['lr'] = 0.01                         # learning rate
args_base6['weight_decay'] = 1e-5
args_base6['batch_size'] = 64
args_base6['epochs'] = 25
args_base6['dropout'] = 0.2
args_base6['output_dim'] = 5                    # number of output classes
args_base6['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args_base6['img_embs_size'] = 128
args_base6['optimizer'] = 'Adam' # 'Adam or SGD'
args_base6['use_scheduler'] = False

args_base7 = dict()                 # batch size for training
args_base7['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args_base7['model_version'] = 'baseline' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args_base7['checkpoint_name'] = 'baseline_5_class_focal_non_augmented_no_scheduler_lr_1e-2_bs_64.pth'
args_base7['data'] = 'non_augmented' # choose between 'augmented' and 'non_augmented'
args_base7['lr'] = 0.01                         # learning rate
args_base7['weight_decay'] = 1e-5
args_base7['batch_size'] = 64
args_base7['epochs'] = 25
args_base7['dropout'] = 0.2
args_base7['output_dim'] = 5                    # number of output classes
args_base7['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args_base7['img_embs_size'] = 128
args_base7['optimizer'] = 'Adam' # 'Adam or SGD'
args_base7['use_scheduler'] = False
args_base7['use_focal_loss'] = True

args_base8 = dict()                 # batch size for training
args_base8['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args_base8['model_version'] = 'baseline' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args_base8['checkpoint_name'] = 'baseline_5_class_ce_non_augmented_no_scheduler_lr_1e-2_bs_64.pth'
args_base8['data'] = 'non_augmented' # choose between 'augmented' and 'non_augmented'
args_base8['lr'] = 0.01                         # learning rate
args_base8['weight_decay'] = 1e-5
args_base8['batch_size'] = 64
args_base8['epochs'] = 25
args_base8['dropout'] = 0.2
args_base8['output_dim'] = 5                    # number of output classes
args_base8['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args_base8['img_embs_size'] = 128
args_base8['optimizer'] = 'Adam' # 'Adam or SGD'
args_base8['use_scheduler'] = False
args_base8['no_class_weights'] = True

def train(args):
    print(args['checkpoint_name'])
    # Build data loader
    if args['data'] == 'augmented':
        train_loader, val_loader, _, class_weights = get_new_data(args['caption_version'], args)
    elif args['data'] == 'non_augmented':
        train_loader, val_loader, _, class_weights = get_data(args['caption_version'], args)

    # Initialize the model for this run
    model_ft = initialize_resnet(args['model_version'], args['img_embs_size'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    class_weights = class_weights.to(device)
    model, optimizer, scheduler, criterion = initialize_model(args, class_weights, train_loader, model_ft)
    model.to(device)
    # TRAINING PIPELINE
    best_val_acc = 0.0
    for e in range(args['epochs']):
        train_loss, val_loss = [], []
        train_acc, val_acc = [], []
        train_err, val_err = [], []
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()
        for i, batch in enumerate(train_loader, 0):
            optimizer.zero_grad()
            data_dict = dict()
            data_dict['caption_and_user_inputs'] = batch[0].float().to(device)
            data_dict['user_input'] = batch[1].float().to(device)
            data_dict['image_inputs'] = batch[2].float().to(device)
            if 'use_bce' in args.keys():
                if args['use_bce'] == True:
                    labels = batch[4].float().to(device).unsqueeze(-1)
                else:
                    labels = batch[4].type(torch.LongTensor).to(device)
            else:
                labels = batch[4].type(torch.LongTensor).to(device)
            outputs = model(data_dict)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if args['use_scheduler']:
                scheduler.step()
            train_loss.append(loss.item())
            if 'use_bce' in args.keys():
                if args['use_bce'] == True:
                    predicted = torch.round(torch.sigmoid(outputs))
                else:
                    _, predicted = torch.max(outputs.data, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader, 0):
                data_dict = dict()
                data_dict['caption_and_user_inputs'] = batch[0].float().to(device)
                data_dict['user_input'] = batch[1].float().to(device)
                data_dict['image_inputs'] = batch[2].float().to(device)
                if 'use_bce' in args.keys():
                    if args['use_bce'] == True:
                        labels = batch[4].float().to(device).unsqueeze(-1)
                    else:
                        labels = batch[4].type(torch.LongTensor).to(device)
                else:
                    labels = batch[4].type(torch.LongTensor).to(device)
                outputs = model(data_dict)
                loss = criterion(outputs, labels)
                val_loss.append(loss.item())
                if 'use_bce' in args.keys():
                    if args['use_bce'] == True:
                        predicted = torch.round(torch.sigmoid(outputs))
                    else:
                        _, predicted = torch.max(outputs.data, 1)
                else:
                    _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total
        if val_acc > best_val_acc:
          # save model
          PATH = os.path.join(args['checkpoint_name'])
          torch.save(model.state_dict(), PATH)
          best_val_acc=val_acc
        print("Epoch [{}/{}], training loss:{:.5f}, validation loss:{:.5f}, train accuracy:{:.5f}, validation accuracy:{:.5f}".format(e + 1, args['epochs'], np.mean(train_loss),np.mean(val_loss), train_acc, val_acc))

# train(args)
# train(args1)
# train(args2)
# train(args3)
# train(args4)
# train(args5)
# train(args6)
# train(args7)
# train(args8)
# train(argsa)

# train(argsn)
# train(args1n)
# train(args2n)
# train(args3n)
# train(args4n)
# train(args5n)
# train(args7n)
# train(args6b)
# train(args6c)
# train(args6d)
# train(argse)
# train(argsf)
# train(argsg)
# train(argsh)
# train(argsi)
# train(argsj)
# train(args_base)
# train(args_base1)
# train(args_base2)
# train(args_base3)
# train(args_base4)
# train(args_base5)
# train(args_base6)
# train(args_base7)
train(args_base8)