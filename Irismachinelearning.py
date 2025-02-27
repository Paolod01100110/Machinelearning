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

# branch 3 visualizzazione dei dati
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='species', y='sepal length (cm)')
plt.title('Boxplot della lunghezza del sepalo per specie')
plt.xlabel('Specie')
plt.ylabel('Lunghezza sepalo (cm)')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', style='species')
plt.title('Scatter plot: Lunghezza sepalo vs Lunghezza petalo per specie')
plt.xlabel('Lunghezza sepalo (cm)')
plt.ylabel('Lunghezza petalo (cm)')
plt.legend(title='Specie')
plt.show()