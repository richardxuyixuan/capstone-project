import os
import numpy as np

import torch
from data_pipeline import get_data
from utils import initialize_resnet, initialize_model
# MODEL SETTINGS
args = dict()                 # batch size for training
args['caption_version'] = 'short'  # 'short' or 'long'
args['model_version'] = 'image_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args['checkpoint_name'] = 'image_only_lr_1e-2_bs_64.pth'

args['lr'] = 0.01                         # learning rate
args['weight_decay'] = 1e-5
args['batch_size'] = 64
args['epochs'] = 25
args['dropout'] = 0.2
args['output_dim'] = 5                    # number of output classes
args['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args['img_embs_size'] = 128
args['optimizer'] = 'Adam' # 'Adam or SGD'

# Build data loader
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
            data_dict['caption_inputs'] = batch[0].float().to(device)
            data_dict['user'] = batch[1].float().to(device)
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
