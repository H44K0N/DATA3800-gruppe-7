import os

import gdown

DATA_PATH = "data/raw_dataset.csv"

# Download the raw data from Google Drive unless it is already here
if not os.path.exists(DATA_PATH):
    os.makedirs("data", exist_ok=True)
    url = "https://drive.google.com/file/d/1uy0j_7LwxPG_a9nXmqvfxGJDMLGpbFHo/view?usp=sharing"
    gdown.download(url, DATA_PATH, quiet=False)
