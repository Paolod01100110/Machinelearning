# Affrontare il problema "Quanto ha piovuto? II" (Kaggle)
# Previsione della pioggia oraria da sequenze radar con LSTM a due strati
#l'obiettivo è Allenare un modello LSTM (Long Short-Term Memory) per prevedere la quantità di pioggia oraria a partire da sequenze temporali di dati radar polarimetrici.
#Il codice:
#Simula un dataset coerente con il problema reale
#Costruisce un modello LSTM profondo (2 strati) per apprendere pattern temporali
#Addestra il modello con tecniche di regolarizzazione
#Valuta la performance con una metrica adeguata (MAE)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential #un tipo di modello feed-forward semplice, dove ogni layer si collega al successivo.
from tensorflow.keras.layers import LSTM, Dense, Dropout 
#LSTM una versione avanzata delle RNN, in grado di gestire dipendenze a lungo termine grazie a gate che controllano cosa memorizzare o dimenticare.
#Dense un  layer completamente connesso, tipico nelle fasi finali di regressione
#Dropout, tecnica di regularizzazione, che disattiva casualmente neuroni durante il training per prevenire overfitting.
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# 1. Generazione di dati sintetici coerenti con la competizione
n_sequences = 1000      # Numero di esempi, abbiamo 1000 esempi/istanze, cioè 1000 eventi radar da prevedere.
timesteps = 30          # Lunghezza di ogni sequenza (temporal window)
n_features = 4          # Es. Reflectivity(misura la quantità di energia riflessa), Zdr, Kdp, RhoHV

# Simuliamo i dati radar (valori casuali ma strutturati)
X = np.random.normal(size=(n_sequences, timesteps, n_features)) #Qui viene creato il dataset X con valori casuali ma strutturati:

# Simuliamo il target Expected: media ponderata + rumore
weights = np.array([0.5, 0.2, 0.2, 0.1]) #È un array di pesi che riflette l'importanza relativa delle 4 feature radar:
rain_intensity = np.tensordot(X, weights, axes=([2], [0])) #Questo è un prodotto scalare su ogni time step, ovvero: Per ogni punto temporale, si calcola una combinazione lineare delle feature.
#Teoricamente, questo rappresenta una stima istantanea dell’intensità di pioggia.
#Risultato: rain_intensity è un array di forma (1000, 30):
#1000 sequenze di intensità di pioggia calcolate per ciascun timestep

y = rain_intensity.mean(axis=1) + np.random.normal(scale=0.1, size=n_sequences)
#Si prende la media temporale della pioggia su ciascuna sequenza.
#Quindi ora otteniamo un solo valore per ogni sequenza: la pioggia oraria totale
#Si aggiunge del rumore gaussiano, per simulare la variabilità naturale e le imprecisioni delle misure radar, questo rende il problema piu realistico
#Stiamo simulando in modo sensato una relazione non perfettamente deterministica tra i dati radar e la pioggia caduta — esattamente come accade nella realtà, dove c’è rumore nei sensori e nella formazione delle precipitazioni.


# 2. Split train/validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Costruzione del modello LSTM a due strati
model = Sequential([  #sequential, Si costruisce un modello a pila, dove ogni layer prende in input l’output del precedente.
    LSTM(64, return_sequences=True, input_shape=(timesteps, n_features)), #64 unità: ogni unità è una cellula LSTM, capace di memorizzare informazioni temporali.
    LSTM(32),
    Dropout(0.2), #Disattiva casualmente il 20% dei neuroni a ogni epoca, al fine di prevenire l'overfitting
    Dense(1)  # Regressione: pioggia oraria prevista, Un layer di output completamente connesso con 1 neurone. estituisce un valore reale continuo → la quantità di pioggia prevista in mm/h.
])

model.compile(optimizer='adam', loss='mae') #Adam è un ottimo algoritmo per il training di reti neurali profonde.

model.summary()

# 4. Training del modello
history = model.fit(
    X_train, y_train, #x sequenze radar, y pioggia oraria simulata
    validation_data=(X_val, y_val), # per monitorare la performance su dati mai visti durante il training
    epochs=20, #Ogni epoca è un passaggio completo su tutti i dati di training, qui diciamo al modello: prova ad apprendere fino a 20 volte, attenzione all'early stopping
    batch_size=32, #i dati non vengono processati uno alla volta ma in mini batch di 32 sequenze
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

# 5. Valutazione
preds = model.predict(X_val).flatten() #Usa il modello allenato per generare previsioni sulla parte di validation, Assicura che l’output sia un array 1D.
mae = mean_absolute_error(y_val, preds)
print(f"MAE sul validation set: {mae:.4f}")

# 6. Visualizzazione risultati
plt.figure(figsize=(10, 5))
plt.plot(y_val, label='Valori Reali')
plt.plot(preds, label='Predetti', alpha=0.7)
plt.legend()
plt.title('Confronto tra pioggia reale e prevista')
plt.xlabel('Campione')
plt.ylabel('Pioggia oraria (simulata)')
plt.grid(True)
plt.tight_layout()
plt.show()
