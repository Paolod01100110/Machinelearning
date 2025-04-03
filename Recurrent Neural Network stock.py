import numpy as np

def create_sequences(data, window_size):
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, concatenate, Embedding, Reshape, BatchNormalization
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import yfinance as yf

# Impostazioni iniziali
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print('Using GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print('Using CPU')

idx = pd.IndexSlice
sns.set_style('whitegrid')
np.random.seed(42)

results_path = Path('results', 'lstm_embeddings')
results_path.mkdir(parents=True, exist_ok=True)

# Download dati da Yahoo Finance
tickers = ['AAPL', 'GOOG']
# Download dei dati da Yahoo Finance
raw_data = yf.download(tickers, start='2015-01-01', end='2025-04-02', interval='1d', group_by='ticker')

# Ricostruisci il DataFrame in formato lungo, con Ticker come colonna esplicita
dfs = []
for ticker in tickers:
    df = raw_data[ticker].copy()
    df['Date'] = df.index
    df['Ticker'] = ticker
    dfs.append(df.reset_index(drop=True))

data = pd.concat(dfs, ignore_index=True)

data['ticker'] = pd.factorize(data['Ticker'])[0]
data['month'] = data['Date'].dt.month
data = pd.get_dummies(data, columns=['month'], prefix='month')
data.info()
full_data = data.copy()


# Impostazioni
window_size = 52
n_tickers = data.ticker.nunique()

# Suddivisione dati
train_data = data[data['Date'] <= '2016-12-31']
test_data = data[data['Date'].dt.year == 2017]

exclude = ['Date', 'Ticker', 'ticker']
sequence = [col for col in train_data.columns if col not in exclude]

X_train, y_train = create_sequences(train_data[sequence].values, window_size)
X_test, y_test = create_sequences(test_data[sequence].values, window_size)

# Cast a float32 per compatibilità con TensorFlow
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# Costruzione modello
K.clear_session()
n_features = len(sequence)

returns = Input(shape=(window_size, n_features), name='Returns')
tickers_input = Input(shape=(1,), name='Tickers')
months_input = Input(shape=(12,), name='Months')

lstm1 = LSTM(units=25, dropout=.2, return_sequences=True, name='LSTM1')(returns)
lstm2 = LSTM(units=10, dropout=.2, name='LSTM2')(lstm1)

ticker_embedding = Embedding(input_dim=n_tickers, output_dim=5, input_length=1)(tickers_input)
ticker_embedding = Reshape(target_shape=(5,))(ticker_embedding)

merged = concatenate([lstm2, ticker_embedding, months_input], name='Merged')
bn = BatchNormalization()(merged)
fc = Dense(10, name='FC1')(bn)
output = Dense(1, name='Output')(fc)

rnn = Model(inputs=[returns, tickers_input, months_input], outputs=output)
rnn.summary()

optimizer = tf.keras.optimizers.Adam()
rnn.compile(loss='mse', optimizer=optimizer)

# Callbacks
lstm_path = (results_path / 'lstm.regression.keras').as_posix()
checkpointer = ModelCheckpoint(filepath=lstm_path, verbose=1, monitor='val_loss', mode='min', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training
training = rnn.fit(
    [
        X_train,
        train_data['ticker'].values[window_size:].reshape(-1, 1).astype(np.float32),
        train_data.filter(like='month').values[window_size:].astype(np.float32)
    ],
    y_train,
    epochs=50,
    batch_size=64,
    validation_data=(
        [
            X_test,
            test_data['ticker'].values[window_size:].reshape(-1, 1).astype(np.float32),
            test_data.filter(like='month').values[window_size:].astype(np.float32)
        ],
        y_test
    ),
    callbacks=[early_stopping, checkpointer],
    verbose=1
)

test_predict = pd.Series(
    rnn.predict([
        X_test,
        test_data['ticker'].values[window_size:].reshape(-1, 1).astype(np.float32),
        test_data.filter(like='month').values[window_size:].astype(np.float32)
    ]).squeeze(),
    index=pd.RangeIndex(start=0, stop=len(y_test))
)

# ✅ Corretto: y_test.squeeze() e test_predict.values sono entrambi 1D

min_len = min(len(y_test), len(test_predict))
df = pd.DataFrame({
    'ret': y_test.ravel()[:min_len],
    'y_pred': test_predict.values.ravel()[:min_len]
})


df['deciles'] = pd.qcut(df.y_pred.rank(method="first"), q=5, labels=False, duplicates='drop')
ic = spearmanr(df.ret, df.y_pred)[0] * 100

df.info()

test_predict.to_frame('prediction').to_hdf(results_path / 'predictions.h5', 'predictions')

rho, p = spearmanr(df.ret, df.y_pred)
print(f'{rho*100:.2f} ({p:.2%})')

# Visualizzazione
fig, axes = plt.subplots(ncols=2, figsize=(14,4))
sns.barplot(x='deciles', y='ret', data=df, ax=axes[0])
axes[0].set_title('Weekly Fwd Returns by Predicted Quintile')
axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
axes[0].set_ylabel('Weekly Returns')
axes[0].set_xlabel('Quintiles')

axes[1].plot(df.index, df.ret.rolling(4).mean(), label='Rolling Returns')
axes[1].axhline(0, c='k', lw=1)
axes[1].set_title(f'4-Week Rolling Returns | IC avg: {ic:.2f}')
axes[1].set_ylabel('Returns')
axes[1].set_xlabel('Index')

sns.despine()
fig.tight_layout()
fig.savefig(results_path / 'lstm_reg')

from datetime import datetime

# Date target per le previsioni
forecast_dates = pd.to_datetime([
    '2025-04-03', '2025-04-04', '2025-04-05', '2025-04-06', '2025-04-07'
])

# Ricontrolla che i ticker testuali siano separati da quelli numerici
if 'ticker_code' not in data.columns:
    data['ticker_code'] = pd.factorize(data['Ticker'])[0]

def forecast_next_days(ticker_name):
    # Cerca righe dove Ticker matcha il nome dato (usa full_data, non data)
    ticker_rows = full_data[full_data['Ticker'].astype(str).str.upper().str.strip() == ticker_name.upper()]
    if ticker_rows.empty:
        raise ValueError(f"Ticker '{ticker_name}' not found in dataset. Trovati: {full_data['Ticker'].unique()}")

    ticker_code = ticker_rows['ticker'].iloc[0]

    # Subset dei dati solo per quel titolo
    ticker_df = full_data[full_data['ticker'] == ticker_code].sort_values('Date')

    if len(ticker_df) < window_size:
        raise ValueError(f"Not enough data to forecast for {ticker_name}")

    # Ultima finestra disponibile
    last_window = ticker_df.iloc[-window_size:][sequence].values.astype(np.float32)
    X_input = last_window.reshape(1, window_size, len(sequence))

    # Input per codice ticker e mese
    ticker_input = np.full((1, 1), ticker_code, dtype=np.float32)
    month_vec = ticker_df.iloc[-1].filter(like='month').values.reshape(1, -1).astype(np.float32)

    predictions = []

    for i in range(5):  # Previsioni per 5 giorni futuri
        y_pred = rnn.predict([X_input, ticker_input, month_vec], verbose=0).squeeze()
        predictions.append(y_pred)

        # Aggiungi la nuova previsione all'input
        new_row = np.zeros((len(sequence),), dtype=np.float32)
        if ticker_name in sequence:
            new_row[sequence.index(ticker_name)] = y_pred
        X_input = np.concatenate([X_input[:, 1:, :], new_row.reshape(1, 1, -1)], axis=1)

    return pd.Series(predictions, index=forecast_dates, name=ticker_name)





# Ottieni previsioni
aapl_preds = forecast_next_days('AAPL')
goog_preds = forecast_next_days('GOOG')

# Plot
plt.figure(figsize=(10, 5))
plt.plot(forecast_dates, aapl_preds, marker='o', label='AAPL Prediction')
plt.plot(forecast_dates, goog_preds, marker='s', label='GOOG Prediction')
plt.title('Previsioni di rendimento: 3-7 Aprile 2025')
plt.xlabel('Data')
plt.ylabel('Rendimento previsto')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(results_path / 'forecast_next_5_days.png')
plt.show()