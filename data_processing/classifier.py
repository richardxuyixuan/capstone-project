from __future__ import annotations
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.models as models
import torchvision.transforms as transforms
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

### Prepare Dataset 
class ClassifierDataset(Dataset):
      """Classifier dataset."""
      def __init__(self, csv_file, root_dir, transform=None):
            """
            Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                  on a sample.
            """
            self.annotation = pd.read_csv(csv_file).reset_index()
            self.root_dir = root_dir
            self.transform = transform
            self.paths,self.labels = [],[]
            for index, row in self.annotation.iterrows():
                  if int(row['class']) == 2:
                        file_name = row['name']
                        data_path = os.path.join(root_dir,file_name)
                        self.labels.append(2)
                  elif int(row['class']) == 1:
                        file_name = row['name']
                        data_path = os.path.join(root_dir,file_name)
                        self.labels.append(1)
                  else:
                        file_name = row['name']
                        data_path = os.path.join(root_dir,file_name)
                        self.labels.append(0)
                  self.paths.append(data_path)

      def __getitem__(self, idx):
            image = Image.open(self.paths[idx]).convert('RGB')
            label = self.labels[idx]
            if self.transform:
                  image = self.transform(image)
            return [self.paths[idx],image,label]

      def __len__(self):
            return len(self.paths)


### Load Dataset
def load_dataset(dataset):
      # Calculate split lengths
      total_size = len(dataset)
      #print(sketch_dataset.size())
      train_size = round(0.7*total_size)
      valid_size = round(0.15*total_size)
      test_size = round(0.15*total_size)
      print(train_size,valid_size,test_size)
      # Seperate into Train, Val and Test sets
      seed = 0
      train_set, valid_set, test_set = random_split(dataset, [train_size,valid_size,test_size], generator=torch.Generator().manual_seed(seed))
      # show the size of each dataset
      print("# Train Set: " + str(len(train_set)))
      print("# Test Set: " + str(len(valid_set)))
      print("# Val Set: " + str(len(test_set)))
      return train_set,valid_set,test_set

class ClassifierFC(nn.Module):
    def __init__(self):
        super(ClassifierFC, self).__init__()
        self.name = 'classifier'
        self.fc1 = nn.Linear(1000, 5000)
        self.fc2 = nn.Linear(5000, 320)
        self.fc3 = nn.Linear(320, 3)

    def forward(self, x):
        x = x.view(-1, 1000)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def evaluate(transfer,model, data_loader, criterion):
      total_val_loss = 0.0
      total_val_err = 0.0
      total_epoch = 0 
      for i, data in enumerate(data_loader, 0):
            # Get the inputs
            paths, inputs, label = data
            inputs = inputs.to(torch.device("cuda"))
            # make changes to the label (one-hot encoding)
            labels = torch.zeros(len(label),3).float()
            labels = labels.to(torch.device("cuda"))
            for j in range(len(label)):
                  index = label[j].item()
                  labels[j][index] = 1.0
      outputs = model(transfer(inputs))
      loss = criterion(outputs, labels.float())
      #select index with maximum prediction score
      pred = outputs.max(1, keepdim=True)[1]
      pred = pred.cpu()
      labels = labels.cpu()
      total_val_err += pred.eq(label.view_as(pred)).sum().item()
      total_val_loss += loss.item()
      total_epoch += len(label)
      val_err = float(total_val_err) / (total_epoch)
      val_loss = float(float(total_val_loss) / (i+1))
      return val_err,val_loss
############################ Training Curve ####################################
def plot_train_val_curve(path):
      """ Plots the training curve for a model run, given the csv files
      containing the train/validation error/loss.

      Args:
            path: The base path of the csv files produced during training
      """
      train_err = np.loadtxt("{}_train_err.csv".format(path))
      val_err = np.loadtxt("{}_val_err.csv".format(path))
      train_loss = np.loadtxt("{}_train_loss.csv".format(path))
      val_loss = np.loadtxt("{}_val_loss.csv".format(path))
      plt.title("Train vs Validation Accuracy")
      n = len(train_err) # number of epochs
      plt.plot(range(1,n+1), train_err, label="Train")
      plt.plot(range(1,n+1), val_err, label="Validation")
      plt.xlabel("Epoch")
      plt.ylabel("Accuracy")
      plt.legend(loc='best')
      plt.show()
      plt.title("Train vs Validation Loss")
      plt.plot(range(1,n+1), train_loss, label="Train")
      plt.plot(range(1,n+1), val_loss, label="Validation")
      plt.xlabel("Epoch")
      plt.ylabel("Loss")
      plt.legend(loc='best')
      plt.show()

def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path

def train(annotation_file, img_dir, batch_size=10,learning_rate=0.01,num_epochs=30):
      # Fixed PyTorch random seed for reproducible result
      torch.manual_seed(1000)
      # load data
      transform =transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
      data = ClassifierDataset(annotation_file,img_dir,transform)
      train_set,valid_set,test_set = load_dataset(data)
      # dataloader 
      train_data_loader = DataLoader(dataset=train_set,batch_size=batch_size, shuffle=True)
      val_data_loader = DataLoader(dataset=valid_set,batch_size=batch_size, shuffle=True)
      test_data_loader = DataLoader(dataset=test_set,batch_size=batch_size, shuffle=True)
      # load model
      transfer = models.resnet50(pretrained=True).to(torch.device("cuda"))
      # Turn on update all params
      for param in transfer.parameters():
            param.requires_grad = True
      net = ClassifierFC().to(torch.device("cuda"))
      # loss function
      criterion = nn.CrossEntropyLoss()
      # set up optimizer, update weights for model and resnet18
      param = [] # put the weights in a list
      for i in transfer.parameters():
            param.append(i)
      for j in net.parameters():
            param.append(j) 
      optimizer = optim.Adam(param, lr=learning_rate)
      # Set up some numpy arrays to store the training/test loss/erruracy
      train_err,train_loss,val_err,val_loss = np.zeros(num_epochs),np.zeros(num_epochs),np.zeros(num_epochs),np.zeros(num_epochs)
      ########################################################################
      # Train the network
      # Loop over the data iterator and sample a new batch of training data
      # Get the output from the network, and optimize our loss function.
      start_time = time.time()
      for epoch in range(num_epochs):  # loop over the dataset multiple times
            total_train_loss = 0.0
            total_train_err = 0.0
            total_epoch = 0
            for i, data in enumerate(train_data_loader, 0):
                  # Get the inputs
                  paths, inputs, label = data
                  inputs = inputs.to(torch.device("cuda"))
                  # make changes to the label (one-hot encoding)
                  labels = torch.zeros(len(label),3).float()
                  for j in range(len(label)):
                        index = label[j].item()
                        labels[j][index] = 1.0
                  labels = labels.to(torch.device("cuda"))
                  # Zero the parameter gradients
                  optimizer.zero_grad()
                  # Forward pass, backward pass, and optimize
                  outputs = net(transfer(inputs))
                  loss = criterion(outputs, labels.float())
                  loss.backward()
                  optimizer.step()
                  # Calculate the statistics
                  #select index with maximum prediction score
                  pred = outputs.max(1, keepdim=True)[1]
                  pred = pred.cpu()
                  labels = labels.cpu()
                  total_train_err += pred.eq(label.view_as(pred)).sum().item()
                  total_train_loss += loss.item()
                  total_epoch += len(labels)
            train_err[epoch] = float(total_train_err) / (total_epoch)
            train_loss[epoch] = float(float(total_train_loss) / (i+1))
            val_err[epoch],val_loss[epoch] = evaluate(transfer,net,val_data_loader, criterion)
            print(("Epoch {}: Train Accuracy: {:.3f}, Train loss: {:.3f}, \n  Validation Accuracy: {:.3f}, Validation loss: {:.3f}\n ").format(
                        epoch + 1,train_err[epoch],train_loss[epoch],val_err[epoch],val_loss[epoch]))
            # Save the current model (checkpoint) to a file
            model_path = get_model_name(net.name, batch_size, learning_rate, epoch+1)
            net_path = get_model_name("resnet50", batch_size, learning_rate, epoch+1)
            
            torch.save(transfer.state_dict(),net_path)
            torch.save(net.state_dict(), model_path)
      print('Finished Training')
      end_time = time.time()
      elapsed_time = end_time - start_time
      print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
      # Write the train/test loss/err into CSV file for plotting later
      epochs = np.arange(1, num_epochs + 1)
      np.savetxt("{}_train_err.csv".format(model_path), train_err)
      np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
      np.savetxt("{}_val_err.csv".format(model_path), val_err)
      np.savetxt("{}_val_loss.csv".format(model_path), val_loss)
      return model_path

def test(annotation_file, img_dir, classifier_path,resnet_path):
      torch.manual_seed(1000)
      preds,gt=[],[]
      # load data
      transform =transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
      data = ClassifierDataset(annotation_file,img_dir,transform)
      trainset,valid_set,test_set = load_dataset(data)
      # dataloader 
      test_data_loader = DataLoader(dataset=test_set,batch_size=16, shuffle=True)
      # load model
      resnet50 = models.resnet50(pretrained=True).to(torch.device("cuda"))
      resnet50.load_state_dict(torch.load(resnet_path))
      classifier = ClassifierFC().to(torch.device("cuda"))
      classifier.load_state_dict(torch.load(classifier_path))
      for i, data in enumerate(test_data_loader, 0):
            paths, inputs,label = data
            inputs = inputs.to(torch.device("cuda"))
            # make changes to the label (one-hot encoding)
            labels = torch.zeros(len(label),3).float()
            for j in range(len(label)):
                  index = label[j].item()
                  labels[j][index] = 1.0
            labels = labels.to(torch.device("cuda"))
            outputs = classifier(resnet50(inputs))
            #select index with maximum prediction score
            pred = outputs.max(1, keepdim=True)[1]
            pred = pred.cpu().squeeze().tolist()
            label = label.cpu().squeeze().tolist()
            #for i in pred:
            #      i = i.tolist()
            inputs = inputs.cpu()
            preds+=pred
            gt+=label
      classes = ['text','text+img','image']
      print("accuracy score: ",accuracy_score(gt, preds))
      cm = confusion_matrix(gt, preds,labels=[0,1,2])
      print(cm)
      fig = plt.figure()
      ax = fig.add_subplot(111)
      cax = ax.matshow(cm)
      plt.title('Confusion matrix of the model\n')
      fig.colorbar(cax)
      plt.xlabel('Predicted labels')
      plt.ylabel('True labels')
      tick_marks = np.arange(len(classes))
      plt.xticks(tick_marks, classes, rotation=45)
      plt.yticks(tick_marks, classes)
      plt.show()


      


if __name__ == '__main__':
      ### to train the model 
      annotation_file = "ads_category.csv"
      img_dir = "raw_imgs"
      model_path = train(annotation_file, img_dir, batch_size=4,learning_rate=0.001,num_epochs=30)
      ### plot train_val_curve
      plot_train_val_curve(model_path)
      ### test model on test dataset 
      resnet_path = "resnet50"
      classifier_path = "fc"
      test(annotation_file, img_dir, classifier_path,resnet_path)