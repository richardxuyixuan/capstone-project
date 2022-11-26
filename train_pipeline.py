import os
import numpy as np
import torch
from data_pipeline import get_new_data
from utils import initialize_resnet, initialize_model

# MODEL SETTINGS
args = dict()                 # batch size for training
args['caption_version'] = 'image'  # 'short', 'long', 'old', 'image', 'image_tuned'
args['model_version'] = 'caption_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args['checkpoint_name'] = 'image_only_clip_lr_1e-2_bs_64.pth'

args['lr'] = 0.01                         # learning rate
args['weight_decay'] = 1e-5
args['batch_size'] = 64
args['epochs'] = 25
args['dropout'] = 0.2
args['output_dim'] = 2                    # number of output classes
args['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args['img_embs_size'] = 128
args['optimizer'] = 'Adam' # 'Adam or SGD'

args1 = dict()                 # batch size for training
args1['caption_version'] = 'image_tuned'  # 'short', 'long', 'old', 'image', 'image_tuned'
args1['model_version'] = 'caption_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args1['checkpoint_name'] = 'image_only_clip_tuned_lr_1e-2_bs_64.pth'

args1['lr'] = 0.01                         # learning rate
args1['weight_decay'] = 1e-5
args1['batch_size'] = 64
args1['epochs'] = 25
args1['dropout'] = 0.2
args1['output_dim'] = 2                    # number of output classes
args1['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args1['img_embs_size'] = 128
args1['optimizer'] = 'Adam' # 'Adam or SGD'

args2 = dict()                 # batch size for training
args2['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args2['model_version'] = 'image_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args2['checkpoint_name'] = 'image_only_resnet50_lr_1e-2_bs_64.pth'

args2['lr'] = 0.01                         # learning rate
args2['weight_decay'] = 1e-5
args2['batch_size'] = 64
args2['epochs'] = 25
args2['dropout'] = 0.2
args2['output_dim'] = 2                    # number of output classes
args2['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args2['img_embs_size'] = 128
args2['optimizer'] = 'Adam' # 'Adam or SGD'

args3 = dict()                 # batch size for training
args3['caption_version'] = 'old'  # 'short', 'long', 'old', 'image', 'image_tuned'
args3['model_version'] = 'caption_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args3['checkpoint_name'] = 'caption_only_not_segmented_image_lr_1e-2_bs_64.pth'

args3['lr'] = 0.01                         # learning rate
args3['weight_decay'] = 1e-5
args3['batch_size'] = 64
args3['epochs'] = 25
args3['dropout'] = 0.2
args3['output_dim'] = 2                    # number of output classes
args3['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args3['img_embs_size'] = 128
args3['optimizer'] = 'Adam' # 'Adam or SGD'

args4 = dict()                 # batch size for training
args4['caption_version'] = 'long'  # 'short', 'long', 'old', 'image', 'image_tuned'
args4['model_version'] = 'caption_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args4['checkpoint_name'] = 'caption_only_segmented_image_lr_1e-2_bs_64.pth'

args4['lr'] = 0.01                         # learning rate
args4['weight_decay'] = 1e-5
args4['batch_size'] = 64
args4['epochs'] = 25
args4['dropout'] = 0.2
args4['output_dim'] = 2                    # number of output classes
args4['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args4['img_embs_size'] = 128
args4['optimizer'] = 'Adam' # 'Adam or SGD'

args5 = dict()                 # batch size for training
args5['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args5['model_version'] = 'caption_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args5['checkpoint_name'] = 'caption_only_segmented_image_pca_lr_1e-2_bs_64.pth'

args5['lr'] = 0.01                         # learning rate
args5['weight_decay'] = 1e-5
args5['batch_size'] = 64
args5['epochs'] = 25
args5['dropout'] = 0.2
args5['output_dim'] = 2                    # number of output classes
args5['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args5['img_embs_size'] = 128
args5['optimizer'] = 'Adam' # 'Adam or SGD'

args6 = dict()                 # batch size for training
args6['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args6['model_version'] = 'early_fusion' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args6['checkpoint_name'] = 'early_fusion_lr_1e-2_bs_64.pth'

args6['lr'] = 0.01                         # learning rate
args6['weight_decay'] = 1e-5
args6['batch_size'] = 64
args6['epochs'] = 25
args6['dropout'] = 0.2
args6['output_dim'] = 2                    # number of output classes
args6['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args6['img_embs_size'] = 128
args6['optimizer'] = 'Adam' # 'Adam or SGD'

args7 = dict()                 # batch size for training
args7['caption_version'] = 'short'  # 'short', 'long', 'old', 'image', 'image_tuned'
args7['model_version'] = 'late_fusion' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args7['checkpoint_name'] = 'late_fusion_lr_1e-2_bs_64.pth'

args7['lr'] = 0.01                         # learning rate
args7['weight_decay'] = 1e-5
args7['batch_size'] = 64
args7['epochs'] = 25
args7['dropout'] = 0.2
args7['output_dim'] = 2                    # number of output classes
args7['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args7['img_embs_size'] = 128
args7['optimizer'] = 'Adam' # 'Adam or SGD'
def train(args):
    print(args['checkpoint_name'])
    # Build data loader
    train_loader, val_loader, _, class_weights = get_new_data(args['caption_version'], args)

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
            labels = batch[3].type(torch.LongTensor).to(device)
            outputs = model(data_dict)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss.append(loss.item())
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
                labels = batch[3].type(torch.LongTensor).to(device)
                outputs = model(data_dict)
                loss = criterion(outputs, labels)
                val_loss.append(loss.item())
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
train(args1)
train(args2)
train(args3)
train(args4)
train(args5)
train(args6)
train(args7)

