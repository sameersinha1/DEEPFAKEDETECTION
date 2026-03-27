# Deepfake Image + Voice Detection (PyTorch)

This project currently includes two scripts:

1. `deepfake_img.py` for image deepfake detection and GAN-based generation.
2. `deepfake_voice.py` for audio deepfake detection with a hybrid model.

## Project Files

- `deepfake_img.py`: End-to-end image pipeline (classification + GAN).
- `deepfake_voice.py`: End-to-end voice pipeline (audio preprocessing + hybrid classifier training).
- `Readme.md`: Project documentation.

## Deepfake Image

### First Pixel Generated

![First generated image output](WhatsApp%20Image%202026-03-27%20at%2020.42.31.jpeg)

### Image Dataset Structure

The image script expects a zip file named `deepfake.zip` in the project root.

After extraction, expected folders are:

```text
dataset/
    deepfake/
        real/
            *.jpg|*.png|...
        Fake/
            *.jpg|*.png|...
```

Notes:
- Folder names are case-sensitive in code: `real` and `Fake`.
- Images are resized to `64x64` during loading.

### What `deepfake_img.py` Does

1. Extracts the image dataset and previews samples.
2. Trains a CNN discriminator to classify Real vs Fake.
3. Evaluates the discriminator (loss, accuracy, confusion matrix).
4. Trains a GAN loop for generation.
5. Saves model checkpoints.

### Image Outputs

- `discriminator_model.pth`: trained discriminator weights.
- `gan_checkpoint.pth`: GAN checkpoint with `epoch`, `G`, `D`, and optimizer states.

## Deepfake Voice

### Voice Pipeline Summary (`deepfake_voice.py`)

The voice script builds a combined dataset from:

- ASVspoof 2019
- Fake-or-Real dataset
- WaveFake dataset

Then it performs:

1. Audio loading (`.wav` and `.flac`).
2. Resampling to `16 kHz`.
3. Mono conversion.
4. Waveform normalization.
5. Fixed-length trim/pad to `3 seconds`.

### Voice Features and Model

- Mel spectrogram branch (CNN):
  - Conv2d(1->16), MaxPool
  - Conv2d(16->32), MaxPool
  - AdaptiveAvgPool to `8x8`
- Handcrafted feature branch:
  - MFCC mean + MFCC std (`13 + 13`)
  - Zero-crossing rate (`1`)
  - Total handcrafted features: `27`
- Hybrid classifier:
  - Concatenate CNN embedding + feature embedding
  - Fully connected binary output with sigmoid

### Voice Training Behavior

- Loss: `BCELoss`
- Optimizer: `Adam` (`lr=0.002`)
- Epochs: `25`
- Batch size: `16`
- Uses gradient clipping (`max_norm=1.0`)
- Prints validation accuracy after training

### Important Note

The current voice script uses hardcoded Kaggle paths. If you run locally, update those paths to your local dataset directories before execution.

## Requirements

Install dependencies before running:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib opencv-python scikit-learn ipython
```

If using GPU, install the CUDA-compatible PyTorch build from the official PyTorch selector.

## How to Run

### Run Image Script

1. Place `deepfake.zip` in the project root.
2. Ensure the zip contains `deepfake/real` and `deepfake/Fake`.
3. Run:

```bash
python deepfake_img.py
```

### Run Voice Script

1. Update dataset paths in `deepfake_voice.py` to match your environment.
2. Run:

```bash
python deepfake_voice.py
```

## Current Notes

- Both scripts are single-file, notebook-style pipelines.
- Paths and most hyperparameters are currently hardcoded.
