import pandas as pd
import os

DATASET_PATH = 'Peugeot.csv'


def load_dataset(dataset_path=DATASET_PATH):
    data = pd.read_excel(dataset_path)
    print(data.head())


load_dataset(dataset_path=DATASET_PATH)
