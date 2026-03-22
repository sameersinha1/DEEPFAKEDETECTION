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

