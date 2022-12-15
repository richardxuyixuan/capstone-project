import os
import numpy as np
import torch
from data_pipeline import get_new_data, get_data
from utils import initialize_resnet, initialize_model

# MODEL SETTINGS
args = dict()
args['caption_version'] = 'short'                               # 1D feature embedding for 5-layer MLP in the 'caption_only' model.
                                                                # Please choose between 'short' for GIT-BERT caption embedding on cropped images with PCA,
                                                                # 'long' for GIT-BERT caption embedding on cropped images without PCA,
                                                                # 'old' for GIT-BERT caption embedding on full images
                                                                # 'image' for CLIP image embedding,
                                                                # 'image_tuned' for fine-tuned-CLIP image embedding.
args['model_version'] = 'early_fusion'                          # model selected for traininng. Please choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion', 'baseline'.
args['checkpoint_name'] = 'captone_adview.pth'                  # checkpoint name to save to the directory during training
args['data'] = 'augmented'                                      # setting for choosing augmented or non-augmented data. Please choose between 'augmented' and 'non_augmented'
args['lr'] = 0.01                                               # learning rate
args['weight_decay'] = 1e-5                                     # weight decay
args['batch_size'] = 64                                         # batch size for training
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

assert args['model_version'] in ['image_only', 'caption_only', 'early_fusion', 'late_fusion'], "Please choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion', 'baseline'."
assert args['caption_version'] in ['short', 'long', 'old', 'image', 'image_tuned'], "Please choose between 'short', 'long', 'old', 'image', 'image_tuned'."
assert '.pth' in args['checkpoint_name'], "Please provide the correct checkpoint name."
assert args['data'] in ['augmented', 'non_augmented'], "Please choose between 'augmented', 'non_augmented'."
assert args['optimizer'] == 'Adam' or args['output_dim'] == 'SGD', "Please choose between Adam or SGD for optimizer."
assert args['output_dim'] in [1, 2, 5], "Please choose from 1, 2, or 5 for output classes."

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

train(args)