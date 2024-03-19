# Mounting google drive in colab
from google.colab import drive

drive.mount('/content/drive')

# Unzip the dataset
!unzip '/content/drive/My Drive/Colab Notebooks/CVproj/archiveData.zip'

import os, glob
#Loading the data_path
file_path = "/content/sign_data/sign_data/"
random_seed = 111

categories = os.listdir(file_path)

#Extrating the list of class_names (or directory names)
name_class = os.listdir(file_path)
name_class

# Data exploration
import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/content/sign_data/sign_data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Import libraries
import time
import copy
import numpy as np
import pandas as pd
from datetime import datetime
from statistics import mean

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import SubsetRandomSampler

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from itertools import combinations

import random
from tqdm import tqdm # Progress Bar

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

#Defining functions for image data visualisation
def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    fig = plt.figure(figsize=(3, 3), facecolor='white')
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='normal',fontweight='normal',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    # Use a grayscale color map
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(2)
    plt.show()


def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Device in use
device

# Load dataset
train_dir = "/content/sign_data/sign_data/train"
test_dir = "/content/sign_data/sign_data/test"
train_csv = "/content/sign_data/sign_data/train_data.csv"
test_csv = "/content/sign_data/sign_data/test_data.csv"

#Data preprocessing - resizing and cnverting to tensor
transforms = transforms.Compose([
    transforms.Resize((100,100)),
    transforms.ToTensor()
])

#Define Siamese Network Dataset class
class SiameseNetworkDataset(Dataset):

    def __init__(self,training_csv=None,training_dir=None,transform=None):
        # used to prepare the labels and images path
        self.training_df=pd.read_csv(train_csv)
        self.training_df.columns =["image1","image2","label"]
        self.training_dir = training_dir
        self.transform = transform

    def __getitem__(self,index):

        # getting the image path
        image1_path=os.path.join(self.training_dir,self.training_df.iat[int(index),0])
        image2_path=os.path.join(self.training_dir,self.training_df.iat[int(index),1])

        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        label = torch.from_numpy(np.array([int(self.training_df.iat[int(index), 2])], dtype=np.float32))

        return img0, img1 , label

    def __len__(self):
        return len(self.training_df)

# Load the train dataset
signature_dataset = SiameseNetworkDataset(train_csv,train_dir,transform=transforms)

# Viewing some image samples
vis_dataloader = DataLoader(signature_dataset,
                           shuffle=True,
                           batch_size=8)
data_iter = iter(vis_dataloader)

example_batch = next(data_iter)
concatenated = torch.cat((example_batch[0], example_batch[1]),0)
imshow(torchvision.utils.make_grid(concatenated))

#Siamese Network Architecture defined
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the CNN Layers
        self.cnn1 = nn.Sequential(

            nn.Conv2d(1, 96, kernel_size=11,stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

            nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),

        )

        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(25600, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),

            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128,2))



    def forward_once(self, x):
        # Forward pass
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2

#Contrastive Loss Function defined
class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

#Splitting Train dataset into train and validation data
from torch.utils.data import SubsetRandomSampler
batch_size = 32

train_dataset = SiameseNetworkDataset(train_csv, train_dir, transform=transforms)

shuffled_idx = torch.randperm(len(train_dataset))
train_idxs = shuffled_idx[:int(0.7*len(shuffled_idx))]
valid_idxs = shuffled_idx[int(0.7*len(shuffled_idx)):]

train_sampler = SubsetRandomSampler(train_idxs)
valid_sampler = SubsetRandomSampler(valid_idxs)

train_dl = DataLoader(train_dataset,batch_size=batch_size, sampler=train_sampler)
valid_dl = DataLoader(train_dataset,batch_size=batch_size, sampler=valid_sampler)

#Size of train data
len(train_dl)

#Size of Val data
len(valid_dl)

#Defining training function
def train_model(model,train_dl, valid_dl,n_epochs, loss_fn, optimizer):

    model.train()
    training_loss_for_plot = []
    valid_loss_for_plot = []
    new_val_loss = []
    new_train_loss = []
    # learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.001, epochs=n_epochs,
                                                steps_per_epoch=len(train_dl))
    print("-----------------------Training-------------------")
    for epoch in range(1, n_epochs+1):
        t0 = datetime.now()
        print(f"Beggining Epoch {epoch}/{n_epochs}...")
        training_loss = []
        valid_loss = []


        for i, data in enumerate(train_dl,0):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)

            output1, output2 = model(img0, img1)
            loss_contrastive = loss_fn(output1, output2, label)

            optimizer.zero_grad()
            loss_contrastive.backward()
            training_loss.append(loss_contrastive.item())
            training_loss_for_plot.append(loss_contrastive.mean().item())
            optimizer.step()
            sched.step()

        with torch.no_grad():
            for i, data in enumerate(valid_dl,0):
                img0, img1, label = data
                img0, img1, label = img0.to(device), img1.to(device), label.to(device)

                output1, output2 = model(img0, img1)
                loss = loss_fn(output1,output2, label)

                valid_loss.append(loss.item())
                valid_loss_for_plot.append(loss.mean().item())
        dt = datetime.now()- t0
        print(f"Training Loss Mean: {mean(training_loss): .5f} | Valid Loss Mean: {mean(valid_loss): .5f} | Time: {dt}")
        new_train_loss.append(mean(training_loss))
        new_val_loss.append(mean(valid_loss))
    return new_train_loss, new_val_loss

#Trainig the model with the define loss function and optimizer for 10 epochs
model = SiameseNetwork()
model = model.to(device)

loss_fn = ContrastiveLoss()
loss_fn = loss_fn.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

#new_train_loss, new_val_loss = train_model(model=model,
                                                        #  train_dl = train_dl,
                                                         # valid_dl = valid_dl,
                                                        #  n_epochs=10,
                                                         # loss_fn=loss_fn,
                                                        #  optimizer=optimizer,
                                                       # )

#Saving the model
#torch.save(model.state_dict(), "SiameseModelWeights.pt")

#Load the Saved Model and displsy its architecture
model.load_state_dict(torch.load("SiameseModelWeights.pt",map_location=torch.device('cpu')))
model.eval()

#Save trainig and validation loss values over each epoch
import pickle


# Save the list to a file
#with open('my_list_train.pkl', 'wb') as file:
   # pickle.dump(new_train_loss, file)

#with open('my_list_val.pkl', 'wb') as file:
   # pickle.dump(new_val_loss, file)

#Load the saved train/val loss values
with open('my_list_train.pkl', 'rb') as file:
    new_train_loss = pickle.load(file)

print(new_train_loss)


with open('my_list_val.pkl', 'rb') as file:
    new_val_loss = pickle.load(file)

print(new_val_loss)

#Display Loss VS Epoch plots

plt.figure(dpi=100)

epoch_arr = range(1,11)
print((np.array(new_train_loss)).shape)
plt.plot(epoch_arr, new_train_loss)
plt.plot(epoch_arr, new_val_loss)
plt.legend(["Training Loss", "Validation Loss"])
plt.grid(False)

# Set the background color of the figure
plt.gca().set_facecolor('white')

# Add an outer border to the figure
border = plt.gca().patch
border.set_edgecolor('black')
border.set_linewidth(1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title("Training and Validation Loss")
plt.show()

#Using the trained model to make predictions on the test data and visualising the results
counter = 0
correct_predictions = 0
test_dl = DataLoader(train_dataset,batch_size=1, sampler=valid_sampler)
for i, data in enumerate(test_dl, 0):
    try:
        x0, x1, label = data
        concatenated = torch.cat((x0, x1), 0)
        output1, output2 = model(x0.to(device), x1.to(device))
        eucledian_distance = F.pairwise_distance(output1, output2)

        # Convert the first element of eucledian_distance to a scalar
        eucledian_distance_scalar = eucledian_distance.item()

        # Perform the comparison based on the tensors
        if torch.all(torch.eq(label, 0)):
            ground_truth_label = "Original"
        else:
            ground_truth_label = "Forged"

        # Predict the label based on the dissimilarity score
        if eucledian_distance_scalar < 1:
            predicted_label = "Original"
        else:
            predicted_label = "Forged"

        # Display the signature pair
        imshow(concatenated[1,:,:],
               'Actual: {} \nPredicted: {}'.format(ground_truth_label, predicted_label))

        # Check if the prediction is correct
        if ground_truth_label == predicted_label:
            correct_predictions += 1
        counter += 1
        if counter > 100:
            break

    except FileNotFoundError as e:
        print(f"File not found: {e.filename}. Skipping to the next iteration.")

#Evaluate Test Accuracy
accuracy = correct_predictions / counter
print("Test Accuracy: {:.2%}".format(accuracy))
