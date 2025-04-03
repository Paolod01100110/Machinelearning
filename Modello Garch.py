import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from scipy.stats import shapiro, jarque_bera, kurtosis, t
from sklearn.model_selection import ParameterGrid
import time

def download_data(ticker, start, end, retries=3, delay=2):
    """Scarica i dati storici con retry automatico."""
    for attempt in range(retries):
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if not data.empty:
                return data
            print(f"Tentativo {attempt + 1}: Dati non disponibili, riprovo...")
        except Exception as e:
            print(f"Errore durante il download: {e}")
        time.sleep(delay)
    raise ValueError("Impossibile scaricare i dati dopo più tentativi.")

#  Scaricare i dati storici
ticker = "AMZN"
start_date = "2015-01-01"
end_date = "2025-03-17"
data = download_data(ticker, start_date, end_date)
prices = data["Close"]
returns = np.log(prices).diff().dropna()

def test_normality(series):
    """Esegue test di normalità."""
    series = series.dropna()
    shapiro_test = shapiro(series)
    jb_test = jarque_bera(series)
    kurt = kurtosis(series, fisher=True).item()
    print(f"Kurtosi: {kurt:.4f}\nShapiro-Wilk Test: p-value = {shapiro_test.pvalue:.4f}\nJarque-Bera Test: p-value = {jb_test[1]:.4f}")
    
    


    # Probability Plot
    sns.histplot(series, kde=True, bins=30)
    plt.title("Distribuzione dei Residui")
    plt.xlabel("Valore del Residuo")
    plt.ylabel("Densità")
    plt.show()
    
    return kurt, jb_test.pvalue

def select_best_garch(train_data, p_values=[1,2,3], q_values=[1,2,3], dist=['normal', 't']):
    """Seleziona il miglior modello GARCH basato su AIC/BIC."""
    best_model, best_params = None, None
    best_score = float("inf")
    for params in ParameterGrid({'p': p_values, 'q': q_values, 'dist': dist}):
        try:
            model = arch_model(train_data, vol='GARCH', p=params['p'], q=params['q'], dist=params['dist'])
            res = model.fit(disp="off")  
            score = res.aic  
            if score < best_score:
                best_score = score
                best_params, best_model = params, res
        except:
            continue
    print(f"Miglior modello: GARCH({best_params['p']}, {best_params['q']}) - {best_params['dist']}")
    return best_model, best_params

def analyze_residuals(residuals):
    """Analizza i residui del modello."""
    kurt_value = kurtosis(residuals.dropna(), fisher=True).item()
    normal_test = test_normality(residuals)
    use_student_t = (kurt_value > 3) and (normal_test[1] < 0.05) #Se la kurtosi è alta (>3) e il p-value del test di normalità è basso (<0.05), i residui non seguono una distribuzione normale.
    return use_student_t #in tal caso, viene scelto un modello con distribuzione t-Student, che meglio cattura valori estremi.

#  Analisi esplorativa preliminare
plt.figure(figsize=(12, 6))
plt.plot(prices, label="Prezzo di Chiusura")
plt.title("Prezzi storici di Amazon")
plt.legend()
plt.show()

plot_acf(returns, lags=40)
plt.title("Autocorrelazione dei Rendimenti")
plt.show()
plot_pacf(returns, lags=40)
plt.title("Autocorrelazione Parziale dei Rendimenti")
plt.show()

plot_acf(returns**2, lags=40)
plt.title("Autocorrelazione della Volatilità")
plt.show()
plot_pacf(returns**2, lags=40)
plt.title("Autocorrelazione Parziale della Volatilità")  
plt.show()

#  Test preliminari sui rendimenti
test_normality(returns)
print("ARCH-LM test p-value:", het_arch(returns)[1])

#  Divisione training e test
Ntest = min(500, len(returns) // 5)  
train, test = returns.iloc[:-Ntest], returns.iloc[-Ntest:]

#  Ottimizzazione GARCH
best_model, best_params = select_best_garch(train)
if best_model is None:
    raise ValueError("❌ Nessun modello GARCH ottimale trovato!")

#  Analisi diagnostica sui residui
residuals = best_model.resid / best_model.conditional_volatility
use_student_t = analyze_residuals(residuals)

#  Riottimizzazione con t-student se necessario
if use_student_t:
    best_model, best_params = select_best_garch(train, dist=['t'])

#  Previsione della volatilità
forecast_horizon = 5
forecast = best_model.forecast(horizon=forecast_horizon)
forecast_volatility = np.sqrt(forecast.variance.iloc[-1].to_numpy())

#  Test di Ljung-Box sui residui e residui quadrati
print("\nTest di Ljung-Box sui residui:")
ljung_resid = acorr_ljungbox(residuals.dropna(), lags=[10], return_df=True) #Controlla l'indipendenza seriale dei residui.
#Se il p-value è basso, significa che ci sono pattern nei residui → il modello non è adeguato.
#Se i residui non seguono una normale, il modello potrebbe sottostimare la probabilità di eventi estremi (crash di mercato, esplosioni di volatilità).
print(ljung_resid)

print("\n Test di Ljung-Box sui residui quadrati:")
ljung_resid_sq = acorr_ljungbox(residuals.dropna()**2, lags=[10], return_df=True)
print(ljung_resid_sq)

#  Visualizzazione
plt.figure(figsize=(10, 5))
plt.plot(range(1, forecast_horizon + 1), forecast_volatility, marker='o', linestyle='dashed', color='red', label='Previsione Volatilità')
plt.title(f"Previsione della Volatilità a {forecast_horizon} giorni con GARCH({best_params['p']},{best_params['q']})")
plt.legend()
plt.grid()
plt.show()

print("Previsioni della volatilità:")
for i in range(forecast_horizon):
    print(f"Giorno {i+1}: {forecast_volatility[i]:.5f}")


