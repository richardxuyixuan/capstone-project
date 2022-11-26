import numpy as np

import torch
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from data_pipeline import get_data, get_new_data
from utils import initialize_resnet, initialize_model
# MODEL SETTINGS
args = dict()                 # batch size for training
args['caption_version'] = 'short'  # 'short' or 'long'
args['model_version'] = 'image_only' # choose between 'image_only', 'caption_only', 'early_fusion', 'late_fusion'
args['checkpoint_name'] = 'image_only_lr_1e-2_bs_64.pth'

args['lr'] = 0.01                         # learning rate
args['weight_decay'] = 1e-5
args['batch_size'] = 1
args['epochs'] = 25
args['dropout'] = 0.2
args['output_dim'] = 2                    # number of output classes
args['hidden_dim'] = [512, 256, 128, 64]  # number of hidden dimensions
args['img_embs_size'] = 128
args['optimizer'] = 'Adam' # 'Adam or SGD'

# Build data loader
train_loader, _, test_loader, class_weights = get_new_data(args['caption_version'], args)

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
        labels = batch[3].type(torch.LongTensor).to(device)
        # load image missing here (to d0)
        outputs = model(data_dict)
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