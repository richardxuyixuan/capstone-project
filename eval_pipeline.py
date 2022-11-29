import numpy as np

import torch
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

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
args['batch_size'] = 1
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
args1['batch_size'] = 1
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
args2['batch_size'] = 1
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
args3['batch_size'] = 1
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
args4['batch_size'] = 1
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
args5['batch_size'] = 1
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
args7['batch_size'] = 1
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
args8['batch_size'] = 1
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
args8_a['batch_size'] = 1
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
args9['batch_size'] = 1
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
argsa['batch_size'] = 1
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
argsb['lr'] = 0.01                         # learning rate
argsb['weight_decay'] = 1e-5
argsb['batch_size'] = 1
argsb['epochs'] = 25
argsb['dropout'] = 0.2
argsb['output_dim'] = 2                    # number of output classes
argsb['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
argsb['img_embs_size'] = 128
argsb['optimizer'] = 'Adam' # 'Adam or SGD'
argsb['use_scheduler'] = True

argsn = dict()                 # batch size for training
argsn['caption_version'] = 'image'  # 'short', 'long', 'old', 'image', 'image_tuned'
argsn['model_version'] = 'caption_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
argsn['checkpoint_name'] = 'image_only_non_augmented_clip_lr_1e-2_bs_64.pth'
argsn['data'] = 'non_augmented' # choose between 'augmented' and 'non_augmented'
argsn['lr'] = 0.001                         # learning rate
argsn['weight_decay'] = 1e-5
argsn['batch_size'] = 1
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
args1n['batch_size'] = 1
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
args2n['batch_size'] = 1
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
args3n['batch_size'] = 1
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
args4n['batch_size'] = 1
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
args5n['batch_size'] = 1
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
args7n['batch_size'] = 1
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
args6b['batch_size'] = 1
args6b['epochs'] = 25
args6b['dropout'] = 0.2
args6b['output_dim'] = 1                    # number of output classes
args6b['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args6b['img_embs_size'] = 128
args6b['optimizer'] = 'Adam' # 'Adam or SGD'
args6b['use_scheduler'] = True
args6b['use_bce'] = True

def eval(args):
    print(args['checkpoint_name'])
    # Build data loader
    if args['data'] == 'augmented':
        train_loader, _, test_loader, class_weights = get_new_data(args['caption_version'], args)
    elif args['data'] == 'non_augmented':
        train_loader, _, test_loader, class_weights = get_data(args['caption_version'], args)

    # Initialize the model for this run
    model_ft = initialize_resnet(args['model_version'], args['img_embs_size'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    class_weights = class_weights.to(device)
    model, optimizer, scheduler, criterion = initialize_model(args, class_weights, train_loader, model_ft)
    model.to(device)

    correct = 0
    total = 0
    y_pred = []
    y_true = []
    model.load_state_dict(torch.load(args['checkpoint_name']))
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(test_loader, 0):
            data_dict = dict()
            data_dict['caption_and_user_inputs'] = batch[0].float().to(device)
            data_dict['user_input'] = batch[1].float().to(device)
            data_dict['image_inputs'] = batch[2].float().to(device)
            if 'use_bce' in args.keys():
                if args['use_bce'] == True:
                    labels = batch[3].float().to(device).unsqueeze(-1)
                else:
                    labels = batch[3].type(torch.LongTensor).to(device)
            else:
                labels = batch[3].type(torch.LongTensor).to(device)
            outputs = model(data_dict)
            if 'use_bce' in args.keys():
                if args['use_bce'] == True:
                    predicted = torch.round(torch.sigmoid(outputs))
                else:
                    _, predicted = torch.max(outputs.data, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_pred.append(predicted.cpu().numpy())
            y_true.append(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total
    y_true= [y_true[i][0] for i in range(len(y_pred))]
    y_pred= [y_pred[i][0] for i in range(len(y_pred))]
    acc = accuracy_score(y_true, y_pred)
    f1_scores = f1_score(y_true, y_pred, average='weighted')

    print("Test accuracy:{:.5f}".format(acc))
    print("F1 score:{:.5f}".format(f1_scores))

    # constant for classes
    classes = ('1','2')
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis], index = [i for i in classes],
                        columns = [i for i in classes])
    ax= plt.subplot()
    sn.heatmap(df_cm, annot=True)
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
    ax.set_title(args['checkpoint_name'] + ' confusion matrix');
    ax.xaxis.set_ticklabels([i for i in classes]); ax.yaxis.set_ticklabels([i for i in classes]);
    plt.savefig(args['checkpoint_name']+'.png')


# eval(args)
# eval(args1)
# eval(args2)
# eval(args3)
# eval(args4)
# eval(args5)
# eval(args6)
# eval(args7)
# eval(args8)
# eval(argsa)
# eval(args8_a)
# eval(argsn)
# eval(args1n)
# eval(args2n)
# eval(args3n)
# eval(args4n)
# eval(args5n)
# eval(args7n)
eval(args6b)