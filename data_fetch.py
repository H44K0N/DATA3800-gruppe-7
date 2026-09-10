import os
import gdown  # pip install gdown (for Google Drive)
import pandas as pd

DATA_PATH = "data/raw_dataset.csv"

# Last ned filen automatisk hvis dere ikke har den lokalt
if not os.path.exists(DATA_PATH):
    os.makedirs("data", exist_ok=True)
    file_id = "DIN_GOOGLE_DRIVE_FILE_ID"
    url = f"https://drive.google.com/file/d/1uy0j_7LwxPG_a9nXmqvfxGJDMLGpbFHo/view?usp=sharing"
    gdown.download(url, DATA_PATH, quiet=False)

# Les inn datasettet
df = pd.read_csv(DATA_PATH)