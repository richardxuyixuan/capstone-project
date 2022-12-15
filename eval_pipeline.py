import numpy as np

import torch
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from data_pipeline import get_new_data, get_data
from utils import initialize_resnet, initialize_model

# MODEL SETTINGS
args = dict()
args['caption_version'] = 'short'                               # 1D feature embedding for 5-layer MLP in the 'caption_only' model.
                                                                # Please choose between 'short' for GIT-BERT caption embedding on cropped images with PCA,
                                                                # 'long' for GIT-BERT caption embedding on cropped images without PCA,
                                                                # 'old' for GIT-BERT caption embedding on full images
                                                                # 'image' for CLIP image embedding, or
                                                                # 'image_tuned' for fine-tuned-CLIP image embedding.
args['model_version'] = 'early_fusion'                          # model selected for traininng. Please choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion', 'baseline'.
args['checkpoint_name'] = 'early_fusion_lr_1e-2_bs_64.pth'      # checkpoint name to load from the directory during evaluation
args['data'] = 'augmented'                                      # setting for choosing augmented or non-augmented data. Please choose between 'augmented' and 'non_augmented'
args['lr'] = 0.01                                               # learning rate
args['weight_decay'] = 1e-5                                     # weight decay
args['batch_size'] = 1                                         # batch size for evaluation
args['epochs'] = 25                                             # number of epochs
args['dropout'] = 0.2                                           # dropout rate
args['output_dim'] = 1                                          # number of output classes (please choose from 1, 2 or 5)
args['hidden_dim'] = [512, 256, 128, 64]                        # number of hidden dimensions in MLP
args['img_embs_size'] = 128                                     # channel size for 2D image embedding output (for 'image_only' and 'early_fusion' models)
args['optimizer'] = 'Adam'                                      # optimizer. Choose between 'Adam or SGD'
args['use_scheduler'] = True                                    # whether or not to use learning scheduler
args['use_bce'] = True                                          # whether or not to use binary cross entropy. If set to false, the model will use weighted cross entropy, which requires the output_dim to be 2 or 5.
args['use_focal_loss'] = False                                  # whether or not to use focal loss. If set to false, the model will use weighted cross entropy. Both options require the output_dim to be 2 or 5.
args['no_class_weights'] = False                                # whether or not to use cross entropy loss. If set to false, the model will use weighted cross entropy. Both options require the output_dim to be 2 or 5.
                                                                # note: only up to one of the loss setting should be set to True at a time
# args['eval_class'] = 0                                        # use this flag if you would like to evaluate the performance on 0: text ads, 1: image ads, 2: image+text ads

assert args['model_version'] in ['image_only', 'caption_only', 'early_fusion', 'late_fusion'], "Please choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion', 'baseline'."
assert args['caption_version'] in ['short', 'long', 'old', 'image', 'image_tuned'], "Please choose between 'short', 'long', 'old', 'image', 'image_tuned'."
assert '.pth' in args['checkpoint_name'], "Please provide the correct checkpoint name."
assert args['data'] in ['augmented', 'non_augmented'], "Please choose between 'augmented', 'non_augmented'."
assert args['optimizer'] == 'Adam' or args['output_dim'] == 'SGD', "Please choose between Adam or SGD for optimizer."
assert args['output_dim'] in [1, 2, 5], "Please choose from 1, 2, or 5 for output classes."


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
            if 'eval_class' in args.keys():
                if batch[3] != args['eval_class']:
                    continue
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
            if 'use_bce' in args.keys():
                if args['use_bce'] == True:
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
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
    if args['output_dim'] == 5:
        classes = ('1', '2', '3', '4', '5')
    else:
        classes = ('1', '2')
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


def roc(args):
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
            if 'eval_class' in args.keys():
                if batch[3] != args['eval_class']:
                    continue
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
            if 'use_bce' in args.keys():
                if args['use_bce'] == True:
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                else:
                    _, predicted = torch.max(outputs.data, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)
            y_pred.append(predicted.cpu().numpy())
            y_true.append(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total
    y_true= [y_true[i][0] for i in range(len(y_pred))]
    y_pred= [y_pred[i][0] for i in range(len(y_pred))]

    # Build roc curve
    prec, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    
    auc_pr = auc(recall, prec)
    auc_roc = auc(fpr, tpr)
    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    pr_display.plot(ax=ax1)
    roc_display.plot(ax=ax2)
    

    ax1.set_title('General Precision-Recall');
    ax2.set_title('General ROC');
    plt.grid()
    plt.savefig(args['checkpoint_name']+'_curve.png')
    print("auc for Precision-Recall: ", auc_pr)
    print("auc for ROC: ", auc_roc)

roc(args)