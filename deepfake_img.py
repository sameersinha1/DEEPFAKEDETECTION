import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
plt.style.use("ggplot")
from IPython.display import Video
from IPython.display import HTML
import torch
print(torch.cuda.is_available())
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from sklearn.metrics import log_loss
log_loss(["Real", "Fake","Fake", "Real"],
            [[0.1, 0.9], [0.9, 0.1], [0.8, 0.2], [0.35, 0.65]])

import zipfile

with zipfile.ZipFile("deepfake.zip", "r") as zip_ref:
    zip_ref.extractall("dataset")

import os

print(os.listdir("dataset"))

print(os.listdir("dataset/deepfake"))
print(len(os.listdir("dataset/deepfake/real")))
print(len(os.listdir("dataset/deepfake/Fake")))
folder = "dataset/deepfake/real"
files = os.listdir(folder)

for file in files[:5]:   # show only first 5 images
    img = cv2.imread(os.path.join(folder, file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.axis("off")
    plt.show()
folder = "dataset/deepfake/Fake"
files = os.listdir(folder)

for file in files[:5]:   # show only first 5 images
    img = cv2.imread(os.path.join(folder, file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.axis("off")
    plt.show()

import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(256*8*8, 1),
            
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D = Discriminator().to(device)

print(D)
# testing the discriminator
import torch
import torch.nn as nn
x = torch.randn(1,3,64,64).to(device)

output = D(x)

print(output)

# this is used for labeling the fake=0 and real =1
real_folder="dataset/deepfake/real"
fake_folder="dataset/deepfake/Fake"

real_files=[os.path.join(real_folder,f) for f in os.listdir(real_folder)]
fake_files=[os.path.join(fake_folder,f) for f in os.listdir(fake_folder)]            

X = real_files + fake_files
y = [1]*len(real_files) + [0]*len(fake_files)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(len(X_train),len(X_test))

from torch.utils.data import Dataset
import torch
import cv2

class DeepfakeDataset(Dataset):

    def __init__(self, files, labels):
        self.files = files
        self.labels = labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        img = cv2.imread(self.files[idx])
        img = cv2.resize(img,(64,64))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        img = torch.tensor(img).permute(2,0,1).float()/127.5-1
        label = torch.tensor(self.labels[idx]).float()

        return img, label
train_dataset = DeepfakeDataset(X_train, y_train)
test_dataset = DeepfakeDataset(X_test, y_test)

#nopw we are doing batch loading 
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)

import os
if os.path.exists("discriminator_model.pth"):
    os.remove("discriminator_model.pth")
    print("old corrupted delted")
else:
    print("no module file found")

import torch
import torch.nn as nn
import torch.optim as optim

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(D.parameters(), lr=0.0002)

epochs = 50
loss_history = []

for epoch in range(epochs):

    epoch_loss = 0

    for images, labels in train_loader:

        # move data to GPU
        images = images.to(device)
        labels = labels.unsqueeze(1).float().to(device)

        # forward pass
        outputs = D(images)

        loss = criterion(outputs, labels)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)

    loss_history.append(epoch_loss)

    print("Epoch", epoch+1, "Loss:", epoch_loss)

#saving ephoc model 
torch.save(D.state_dict(),"discriminator_model.pth")
print("model saved sucessfully")

D=Discriminator().to(device)
D.load_state_dict(torch.load("discriminator_model.pth"))
D.eval()

import torch 
torch.cuda.empty_cache()
torch.cuda.synchronize()

D.eval()

correct = 0
total = 0
test_loss = 0

all_preds = []
all_labels = []

criterion = nn.BCEWithLogitsLoss()

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.unsqueeze(1).float().to(device)

        outputs = D(images)

        loss = criterion(outputs, labels)
        test_loss += loss.item()

        predicted = (torch.sigmoid(outputs) > 0.5).float()

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = correct / total
test_loss = test_loss / len(test_loader)
print("Test Loss:", test_loss)
print("Test Accuracy:", accuracy)

plt.figure(figsize=(8,5))

plt.plot(loss_history, label="Training Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")

plt.legend()
plt.grid(True)

plt.show()

cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Fake", "Real"])

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")
plt.show()

#here we freezing the Discriminator
for param in D.parameters():
    param.requires_grad=False

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(

            nn.Linear(100, 512*4*4),
            nn.ReLU(True),

            nn.Unflatten(1, (512,4,4)),

            nn.ConvTranspose2d(512,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64,3,4,2,1),
            nn.Tanh()
        )

    def forward(self,x):
        return self.model(x)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
D=Discriminator().to(device)
D.load_state_dict(torch.load("discriminator_model.pth"))
D.train()
print("Discriminator loaded")

import torch.nn as nn
criterion=nn.BCEWithLogitsLoss()

import torch.optim as optim
optimizer_G=optim.Adam(G.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizer_D=optim.Adam(D.parameters(),lr=0.0001,betas=(0.5,0.999))

noise_dim=100 #noise dimension

z = torch.randn(1, noise_dim).to(device)

fake_image = G(z)

print(fake_image.shape)
import matplotlib.pyplot as plt
img=fake_image[0].detach().cpu().permute(1,2,0)
plt.imshow(img)
plt.axis("off")
plt.show()

import os

if os.path.exists("gan_checkpoint.pth"):
    os.remove("gan_checkpoint.pth")
    print("Old GAN checkpoint deleted")
else:
    print("No checkpoint found")

epochs = 600
noise_dim = 100

start_epoch = 0

# Try loading checkpoint
try:
    checkpoint = torch.load("gan_checkpoint.pth")

    G.load_state_dict(checkpoint["G"])
    D.load_state_dict(checkpoint["D"])

    optimizer_G.load_state_dict(checkpoint["optimizer_G"])
    optimizer_D.load_state_dict(checkpoint["optimizer_D"])

    start_epoch = checkpoint["epoch"] + 1

    print(f"Resuming training from epoch {start_epoch}")

except:
    print("Starting training from scratch")


fixed_noise = torch.randn(16, noise_dim).to(device)


def show_generated_images(epoch):

    with torch.no_grad():
        fake_images = G(fixed_noise).detach().cpu()

    fake_images = (fake_images + 1) / 2

    plt.figure(figsize=(6,6))

    for i in range(16):

        plt.subplot(4,4,i+1)

        img = fake_images[i].permute(1,2,0)

        plt.imshow(img)
        plt.axis("off")

    plt.suptitle(f"Generated Images Epoch {epoch}")
    plt.show()


for epoch in range(start_epoch, epochs):

    for real_images, _ in train_loader:

        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # Label smoothing
        real_labels = torch.ones(batch_size,1).to(device) * 0.9
        fake_labels = torch.zeros(batch_size,1).to(device)

        # Generate fake images
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_images = G(noise)

        # -----------------------
        # Train Discriminator
        # -----------------------

        real_loss = criterion(D(real_images), real_labels)
        fake_loss = criterion(D(fake_images.detach()), fake_labels)

        loss_D = real_loss + fake_loss

        optimizer_D.zero_grad()
        loss_D.backward()

        # Gradient clipping (prevents exploding loss)
        torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)

        optimizer_D.step()

        # -----------------------
        # Train Generator
        # -----------------------

        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_images = G(noise)

        outputs = D(fake_images)

        loss_G = criterion(outputs, real_labels)

        optimizer_G.zero_grad()
        loss_G.backward()

        torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)

        optimizer_G.step()


    print(f"Epoch {epoch} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

    # Show images every 10 epochs
    if epoch % 10 == 0:
        show_generated_images(epoch)

    # Save checkpoint
    torch.save({
        "epoch": epoch,
        "G": G.state_dict(),
        "D": D.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict()
    }, "gan_checkpoint.pth")
