import pmdarima as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import scipy.stats as stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import boxcox

# Scaricare i dati dello S&P 500 da Yahoo Finance
df = yf.download('^GSPC', start='2015-01-01', end='2025-01-17')
df['Name'] = 'SP500'

# Trasformazione logaritmica dei rendimenti
df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
df.dropna(inplace=True)

goog = df['Log_Returns']

# Test di trasformazione Box-Cox (prima di ACF/PACF)
goog_boxcox, lambda_ = boxcox(goog - goog.min() + 1)
plt.figure(figsize=(6,4))
plt.hist(goog_boxcox, bins=30, alpha=0.6, color='b')
plt.title("Distribuzione dopo trasformazione Box-Cox")
plt.show()
print("Lambda di Box-Cox:", lambda_)

# Test di stazionarietà ADF
def adf_test(series):
    result = adfuller(series)
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])
    if result[1] <= 0.05:
        print("La serie è stazionaria.")
    else:
        print("La serie non è stazionaria.")

print("\nAnalisi di stazionarietà (ADF Test) per la serie dei rendimenti logaritmici:")
adf_test(goog_boxcox)

# Test di eteroschedasticità Breusch-Pagan
X = np.column_stack((np.ones(len(goog_boxcox)), np.arange(len(goog_boxcox))))
bp_test = het_breuschpagan(goog_boxcox, X)
print("\nTest di Breusch-Pagan per l'eteroschedasticità:")
print("Statistiche di test:", bp_test[0], "p-value:", bp_test[1])
if bp_test[1] > 0.05:
    print("Non vi è evidenza di eteroschedasticità.")
else:
    print("Presenza di eteroschedasticità nei residui.")

# Analisi ACF/PACF
plt.figure(figsize=(12, 5))
plt.subplot(121)
plot_acf(goog_boxcox, ax=plt.gca())
plt.title("Autocorrelation Function (ACF)")
plt.subplot(122)
plot_pacf(goog_boxcox, ax=plt.gca())
plt.title("Partial Autocorrelation Function (PACF)")
plt.show()

Ntest = 30
train = goog_boxcox[:-Ntest]
test = goog_boxcox[-Ntest:]

# Modello ARIMA con ricerca ottimizzata
model_arima = pm.auto_arima(train, error_action='ignore', trace=True,
                            suppress_warnings=True, maxiter=20, seasonal=False,
                            stepwise=True, max_p=5, max_q=5, max_d=2,
                            information_criterion='bic', n_jobs=-1)

# Modello SARIMA con ricerca ottimizzata
model_sarima = pm.auto_arima(train, seasonal=True, m=12, 
                             start_p=0, max_p=5, 
                             start_q=0, max_q=5, 
                             start_P=0, max_P=3,
                             start_Q=0, max_Q=3,
                             stepwise=True, suppress_warnings=True, 
                             trace=True, information_criterion='bic', n_jobs=-1)

# Confronto AIC/BIC tra ARIMA e SARIMA
print("AIC ARIMA:", model_arima.aic(), "| BIC ARIMA:", model_arima.bic())
print("AIC SARIMA:", model_sarima.aic(), "| BIC SARIMA:", model_sarima.bic())

# Calcolo metriche di errore
def calculate_errors(actual, predicted):
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    mae = np.mean(np.abs(predicted - actual))
    mape = np.mean(np.abs((predicted - actual) / actual)) * 100
    return rmse, mae, mape

# Previsioni out-of-sample
predictions = {
    "ARIMA": model_arima.predict(n_periods=Ntest),
    "SARIMA": model_sarima.predict(n_periods=Ntest)
}

# Stampa delle metriche di errore
print("\nMetriche di errore:")
for model_name, pred in predictions.items():
    rmse, mae, mape = calculate_errors(test, pred)
    print(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")

# Analisi dei residui
def analyze_residuals(model, train, model_name):
    residuals = train - model.predict_in_sample()
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f"QQ-Plot dei residui - {model_name}")
    plt.subplot(122)
    plt.hist(residuals, bins=30, density=True, alpha=0.6, color='g')
    plt.title(f"Distribuzione dei residui - {model_name}")
    plt.show()

    jb_test = stats.jarque_bera(residuals)
    print(f"\nTest di Jarque-Bera per i residui di {model_name}:")
    print("Statistiche:", jb_test[0], "p-value:", jb_test[1])
    if jb_test[1] > 0.05:
        print("I residui seguono una distribuzione normale.")
    else:
        print("I residui NON seguono una distribuzione normale.")

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plot_acf(residuals, ax=plt.gca())
    plt.title(f"ACF dei residui - {model_name}")
    plt.subplot(122)
    plot_pacf(residuals, ax=plt.gca())
    plt.title(f"PACF dei residui - {model_name}")
    plt.show()

# Analisi dei residui per entrambi i modelli
analyze_residuals(model_arima, train, "ARIMA")
analyze_residuals(model_sarima, train, "SARIMA")

# Convertire 'test' in una Pandas Series con indice temporale
test_series = pd.Series(test, index=df.index[-Ntest:])

# Stampa delle previsioni di rendimento per i prossimi 5 giorni
print("\nPrevisioni di rendimento per i prossimi 5 giorni:")
for model_name, pred in predictions.items():
    print(f"{model_name}: {pred[:5]}")  # Stampiamo solo i primi 5 giorni

# Grafico delle previsioni di rendimento per ciascun modello
plt.figure(figsize=(10, 5))
plt.plot(test_series.index, test_series, marker='o', label="Rendimenti Reali", linestyle='dashed')

for model_name, pred in predictions.items():
    pred_series = pd.Series(pred, index=test_series.index)  # Manteniamo i rendimenti
    plt.plot(pred_series.index, pred_series, marker='s', label=f"Previsioni {model_name}", linestyle='solid')

plt.xlabel("Data")
plt.ylabel("Rendimento previsto")
plt.title("Previsioni dei rendimenti a 5 giorni")
plt.legend()
plt.grid()
plt.show()



