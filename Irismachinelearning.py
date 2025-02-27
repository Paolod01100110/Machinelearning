import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Caricamento del dataset Iris
iris = load_iris()

# Creazione del DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# branch 1: esplorazione iniziale
print("\n### dimensioni del dataset ###")
print(df.shape)

print("\n### tipi di dati ###")
print(df.dtypes)

print("\n### valori mancanti ###")
print(df.isnull().sum())

print("\n### statistiche descrittive iniziali ###")
print(df.describe())

print("\n### conteggio delle specie ###")
print(df['species'].value_counts())

# branch 2 analisi approfondita
print("\n### matrice di correlazione ###")
print(df.corr(numeric_only=True))

print("\n### statistiche per specie ###")
print(df.groupby('species').agg(['mean', 'max', 'min', 'std']))