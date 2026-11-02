import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Paraemeters Setting
LearningRate = 0.1
InitialWeght = 0.5
Epochs = 5
Data =  pd.read_csv("iris_data.csv")

FEATURES = ["x1", "x2", "x3", "x4"]
WEIGHTS = ["bias", "teta1", "teta2", "teta3", "teta4"]
LABELS = {"Iris-setosa": 0, "Iris-versicolor": 1}
print (Data)

def load_data():
    # Load the dataset
    data = pd.read_csv("iris_data.csv")
    # Map the labels to numerical values
    data['target'] = data['label'].map(LABELS)
    return data

def splitData(df, train_per_class=40):
    groups = df.groupby('target', sort=True)
    train = pd.concat([g.iloc[:train_per_class] for _, g in groups], ignore_index=True)
    val = pd.concat([g.iloc[train_per_class:] for _, g in groups], ignore_index=True)
    return train, val
