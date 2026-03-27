import os
import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
def collect_audio_files(base_path):
    audio_files = []
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith((".wav", ".flac")):
                audio_files.append(os.path.join(root, file))
    
    return audio_files

asv_path = "/kaggle/input/datasets/awsaf49/asvspoof-2019-dataset"
for_path = "/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset"
wavefake_path = "/kaggle/input/datasets/walimuhammadahmad/fakeaudio"
asv_files = collect_audio_files(asv_path)
for_files = collect_audio_files(for_path)
wavefake_files = collect_audio_files(wavefake_path)

import os

base = "/kaggle/input/datasets/awsaf49"

for root, dirs, files in os.walk(base):
    for file in files:
        if "cm.train" in file.lower():
            print(os.path.join(root, file))

def parse_asv_labels(protocol_file):
    label_dict = {}

    if not os.path.isfile(protocol_file):
        raise FileNotFoundError(f"File not found: {protocol_file}")

    with open(protocol_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            
            # Skip malformed lines
            if len(parts) < 5:
                continue
            
            file_id = parts[1]
            label = parts[-1].lower()
            
            label_dict[file_id] = 0 if label == "bonafide" else 1

    return label_dict


protocol_file = "/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"

asv_labels = parse_asv_labels(protocol_file)

print("Total labels:", len(asv_labels))

# Preview few samples
for i, (k, v) in enumerate(asv_labels.items()):
    print(k, "->", v)
    if i == 4:
        break

def get_for_label(path):
    parts = path.lower().split("/")
    
    if "fake" in parts:
        return 1
    elif "real" in parts:
        return 0
    else:
        return None

dataset = []

# ASV (only train subset)
asv_train_path = "/kaggle/input/datasets/awsaf49/asvspoof-2019-dataset/LA/LA/ASVspoof2019_LA_train"
asv_files = collect_audio_files(asv_train_path)

for file in asv_files:
    file_id = os.path.basename(file).replace(".flac", "")
    if file_id in asv_labels:
        dataset.append((file, asv_labels[file_id]))

# Fake-or-Real
for file in for_files:
    label = get_for_label(file)
    if label is not None:
        dataset.append((file, label))

# WaveFake
for file in wavefake_files:
    dataset.append((file, 1))

real = sum(1 for _, l in dataset if l == 0)
fake = sum(1 for _, l in dataset if l == 1)

print("Real:", real)
print("Fake:", fake)

train_data, val_data = train_test_split(
    dataset, test_size=0.2, random_state=42,
    stratify=[l for _, l in dataset]
)
import torch.nn.functional as F

def preprocess_audio(file_path, target_len=16000*3):  # 3 sec
    wave, sr = torchaudio.load(file_path)

    # Resample
    if sr != 16000:
        resample = torchaudio.transforms.Resample(sr, 16000)
        wave = resample(wave)

    # Convert to mono
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)

    # Normalize 
    wave = wave / (wave.abs().max() + 1e-9)

    #  FIX LENGTH 
    if wave.shape[1] > target_len:
        wave = wave[:, :target_len]  # trim
    else:
        pad_size = target_len - wave.shape[1]
        wave = F.pad(wave, (0, pad_size))  # pad

    return wave

def extract_features(wave, sr=16000):
    mfcc = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=13)(wave)

    mfcc_mean = mfcc.mean(dim=2)
    mfcc_std = mfcc.std(dim=2)

    zcr = (wave[:, 1:] * wave[:, :-1] < 0).float().mean()

    features = []
    features.extend(mfcc_mean.squeeze().tolist())
    features.extend(mfcc_std.squeeze().tolist())
    features.append(zcr.item())

    return torch.tensor(features)

class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=64
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]

        wave = preprocess_audio(file_path)

        mel = self.mel_transform(wave)
        mel = torch.log(mel + 1e-9)

        features = extract_features(wave)

        return mel, features, torch.tensor(label, dtype=torch.float32)
    
train_dataset = AudioDataset(train_data[:20000])
val_dataset = AudioDataset(val_data[:5000])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten()
        )

        self.fc_features = nn.Sequential(
            nn.Linear(27, 64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(32*8*8 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, mel, features):
        x1 = self.cnn(mel)
        x2 = self.fc_features(features)
        x = torch.cat((x1, x2), dim=1)
        return self.classifier(x)
    
model = HybridModel().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

num_epochs = 25

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for mel, features, labels in train_loader:
        mel = mel.to(device)
        features = features.to(device)
        labels = labels.unsqueeze(1).float().to(device)

        optimizer.zero_grad()

        outputs = model(mel, features)   # NO sigmoid here
        loss = criterion(outputs, labels)

        # Safety check
        if torch.isnan(loss):
            print(f"NaN loss at epoch {epoch+1}, stopping...")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Avg Loss: {avg_loss:.4f}")

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for mel, features, labels in val_loader:
        mel = mel.to(device)
        features = features.to(device)
        labels = labels.unsqueeze(1).to(device)

        outputs = model(mel, features)
        preds = (outputs > 0.5).float()

        correct += (preds == labels).sum().item()
        total += labels.size(0)

print("Accuracy:", correct / total)

