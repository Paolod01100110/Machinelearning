import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Percorso del file CSV
file_path = r"C:\Users\paolo\Desktop\train.csv"

# Caricamento dei dati nel DataFrame
df = pd.read_csv(file_path)

#Le prime righe del DataFrame 
print(df.head())

# Controllare le info
print(df.info())

print(df.describe())

# Controllo i nulli
print(df.isnull().sum())

# STAMPO SONO QUELLI CON CON NULLI >0
print("STAMPO SONO QUELLI CON CON NULLI >0")
print(df.isnull().sum()[df.isnull().sum()>0])

# Sostituisce i volori nulli di LotFrontage (Metri del lotto fronte strada) con 0.
df.fillna({"LotFrontage": 0}, inplace=True)
#missing_values_LotFrontage = df["LotFrontage"].isnull().sum()
#print(f"Numero di valori nulli in LotFrontage dopo la sostituzione: {missing_values_LotFrontage}")


# "Grvl": vicolo giaiato (costo medio), "Pave": vicolo pavimentato (costo più alto), "NoAlley" : nessun vicolo privato (costo inferiore) 

df.fillna({"Alley": "NoAlley" }, inplace=True) #sostituisce i valori nulli con noalley
#print(df["Alley"].head(50))
#print(df["Alley"].value_counts())

#missing_values_Alley = df["Alley"].isnull().sum()
#print(f"Numero di valori nulli in Alley dopo la sostituzione: {missing_values_Alley}")

# Stone (più costo ), BrkFace (mattoni a vista) un po' meno costoso, BrkCmn (mattoni comuni) ancora meno costoso, 
# valori  nulli i meno costosi di tutti non c'è un rivestimento in materiale esterno
df.fillna({"MasVnrType" : "None"}, inplace=True)
#print(df["MasVnrType"].value_counts())

# l'area di rivestimento in muratura
# DA CONTROLLARE DOPO SE DOVE HO 0 ANCHE IN MasVnrType HO 0
df.fillna({"MasVnrArea": 0}, inplace=True)
#missing_values_MasVnrArea = df["MasVnrArea"].isnull().sum()
#print(f"Numero di valori nulli in MasVnrArea dopo la sostituzione:{missing_values_MasVnrArea}")


# BsmtQual = qualità seminterrato (legato alla sua altezza)
# Classifica crescente costo: Nan, Po , Fa,  TA, Gd,   Ex
df.fillna({"BsmtQual": "NoBsmt"}, inplace=True)
#print(df["BsmtQual"].value_counts())

# BsmtCond = Condizione seminterrato (come BsmtQual) 
#print(df["BsmtCond"].value_counts())
df.fillna({"BsmtCond": "NoBsmt"}, inplace=True)
#print(df["BsmtCond"].value_counts())
#missing_values_BsmtCond = df["BsmtCond"].isnull().sum()
#print(f"Numero di valori nulli in BsmtCond dopo la sostituzione:{missing_values_BsmtCond}")

#BsmtExposure indica il livello di esposizione del seminterrato all’esterno
#print(df["BsmtExposure"].value_counts())
df.fillna({"BsmtExposure": "NoBsmt"}, inplace=True)
#print(df["BsmtExposure"].value_counts())

# BsmtFinType1 indica il tipo principale di finitura del seminterrato, ovvero quanto è stato rifinito e utilizzabile come spazio abitabile.
df.fillna({"BsmtFinType1": "NoBsmt"}, inplace=True)
#print(df["BsmtFinType1"].value_counts())


# BsmtFinType2 rappresenta il secondo tipo di finitura presente nel seminterrato.
#print(df["BsmtFinType2"].value_counts())
df.fillna({"BsmtFinType2": "NoBsmt"}, inplace=True)


# Electrical indica il tipo di impianto elettrico principale installato nella casa.
#print(df["Electrical"].value_counts())
# Sostituiamo con la moda, tanto c'è solo un valore null 
df.fillna({"Electrical": df["Electrical"].mode()[0]}, inplace=True)
#print(df["Electrical"].value_counts())

# FireplaceQu = Qualità del caminetto
#print(df["FireplaceQu"].value_counts())
df.fillna({"FireplaceQu": "NoFireplace"}, inplace=True)
#print(df["FireplaceQu"].value_counts())

# GarageType = Tipo di Garage
#print(df["GarageType"].value_counts())
df.fillna({"GarageType": "NoGarage"}, inplace=True)
#print(df["GarageType"].value_counts())

# GarageYrBlt = anno di costruzione del garage
# Lo elimiano, non ci interessa se il garage è stato costruito prima o dopo.
df.drop(columns=["GarageYrBlt"], inplace=True)
#print(df.shape) # Colonna eliminata


# GarageFinish = ovvero se le pareti e il soffitto del garage sono stati 
# rifiniti con materiali (come cartongesso o vernice) o se sono rimasti grezzi.
df.fillna({"GarageFinish": "NoGarage"}, inplace=True)
#print(df["GarageFinish"].value_counts())


# GarageQual =  Qualità del Garage, rappresenta la qualità complessiva della costruzione del garage, 
# valutata in base ai materiali e alle condizioni strutturali.
df.fillna({"GarageQual": "NoGarage"}, inplace=True)
#print(df["GarageFinish"].value_counts())

# GarageCond rappresenta la condizione generale del garage, 
# ovvero lo stato attuale della struttura e il livello di manutenzione.

df.fillna({"GarageCond": "NoGarage"}, inplace=True)
#print(df["GarageCond"].value_counts())

# PoolQC = qualità Piscina
# se non hanno valori, mettiano 0
df.fillna({"PoolQC": "NoPool"}, inplace=True)
#print(df["PoolQC"].value_counts())

# Fence = Fence rappresenta il tipo di recinzione presente nella proprietà, 
# indicando il livello di privacy e sicurezza fornito.
df.fillna({"Fence": "NoFence"}, inplace=True)
#print(df["Fence"].value_counts())

# MiscFeature = Caratteristiche extra
#print(df["MiscFeature"].value_counts()) # Shed ,Gar2 , Othr, TenC

#misc_feature_counts = df["MiscFeature"].value_counts()
misc_price_impact = df.groupby("MiscFeature")["SalePrice"].median()

# Grafico per capire come mappare
# TOGLI COMMENTO
# plt.figure(figsize=(8,5))
# plt.bar(misc_price_impact.index, misc_price_impact.values, color='seagreen')
# plt.xlabel("Caratteristiche Extra (MiscFeature)")
# plt.ylabel("Prezzo Mediano di Vendita ($)")
# plt.title("Impatto delle Caratteristiche Extra sul Prezzo delle Case")
# plt.xticks(rotation=45)
# plt.show()

# ordiniamo in base ai risultati del grafico :  TenC, Gar2 , Shed , Othr.
df.fillna({"MiscFeature": "NoFeature"}, inplace=True)
#misc_feature_counts = df["MiscFeature"].value_counts()
#print(misc_feature_counts)

# Facciamo il controllo sui valori nulli nuovamente
print("COLONNE NULLE: ")
print(df.isnull().sum()[df.isnull().sum()>0])
# NESSUNA COLONNA HA VALORI MANCANTI

#print(df.info())





# # MATRICE DI CORRELAZIONE CON I VALORI NUMERICI
# correlation_matrix = df.select_dtypes(include=["number"]).corr()
# correlation_matrix
# plt.figure(figsize=(12,8))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
# plt.show()

# trasforzaione delle colonne in categorie ove possibile
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category')

# Sul Quartiere facciamo il get_dummies
#df = pd.get_dummies(df, columns=['Neighborhood'], drop_first=True)
#print(df.head())

print(df.describe) 
#---------------------MODELLO-----

X = df.drop(columns=['SalePrice'])
y = df['SalePrice']


# Dividiamo il dataset in training (80%) e test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  4️⃣ Ottimizzazione degli Iperparametri con GridSearchCV
# =========================
# Definiamo i parametri da ottimizzare
param_grid = {
    'n_estimators': [50, 100, 200],  # Numero di alberi
    'learning_rate': [0.01, 0.1, 0.2],  # Velocità di apprendimento
    'max_depth': [3, 5, 7],  # Profondità massima degli alberi
    'tree_method': ["hist"],  # REQUIRED for categorical support
    'enable_categorical': [True],  # Enables categorical handling
    'lambda': [1],  # Regolarizzazione L2 (ridge)
    'alpha': [0.5]   # Regolarizzazione L1 (lasso)
}


# Inizializziamo il modello
xgb_regressor_grid = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

# Eseguiamo GridSearchCV per trovare la combinazione migliore
grid_search = GridSearchCV(xgb_regressor_grid, param_grid, cv=3, scoring="r2", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Miglior set di parametri trovato
best_params = grid_search.best_params_
print(best_params)


# =========================
#  Addestriamo XGBoost con i Migliori Parametri
# =========================
best_xgb = xgb.XGBRegressor(**best_params, objective="reg:squarederror", random_state=42)
best_xgb.fit(X_train, y_train)

# Facciamo previsioni
y_pred_best = best_xgb.predict(X_test)

# Calcoliamo le metriche aggiornate
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred_best))
r2_best = r2_score(y_test, y_pred_best)

print(f"RMSE: {rmse:.2f}")
print(f"Migliore R2: {r2_best}")

# Visualizza l'importanza delle feature
xgb.plot_importance(best_xgb)
plt.title("Feature Importance")
plt.show()
#questo grafico misura il peso delle caratteristiche in base a quante volte viene usata la caratteristica come condizione per dividere il nodo (la frequenza dell'utilizzo della feature nei nodi)
#Neighborhood è una variabile categoriale con molte classi, ogni classe può essere usata in diversi split, è importante per avere una predizione precisa, perchè viene usato molto per formare i nodi degli alberi decisionali
#Le variabili categoriali con molte categorie tendono a essere utilizzate più spesso nei nodi.
#Anche se Neighborhood è usato spesso negli split, il suo effetto sul prezzo non è così forte.
#Altre feature come GrLivArea, OverallQual e TotalBsmtSF hanno un impatto maggiore sul prezzo.
#Le feature numeriche tendono ad avere un impatto più forte sulle previsioni rispetto alle variabili categoriali.


feature_importance = grid_search.best_estimator_.feature_importances_
features = X_train.columns


#Ordina le feature per importanza
sorted_idx = np.argsort(feature_importance)[::-1]
plt.figure(figsize=(20, 5))
plt.bar(range(len(sorted_idx[:20])), feature_importance[sorted_idx[:20]], align="center")
plt.xticks(range(len(sorted_idx[:20])), [features[i] for i in sorted_idx[:20]], rotation=45)
plt.title("Top 20 Feature Importance")
plt.show()
#questo grafico misura quanto la caratteristica fa variare il prezzo della casa (sono proprio i coefficienti della regressione), indica l'importanza basata sulla varianza della predizione
#Se vuoi capire come il modello prende decisioni, usa il primo grafico.

#Se vuoi capire quali variabili influenzano di più il prezzo, usa il secondo grafico

# Selezionare le 20 feature più importanti
sorted_idx = np.argsort(feature_importance)[::-1][:20] #estituisce gli indici che ordinano l'array feature_importance in ordine crescente
top_features = [features[i] for i in sorted_idx]



# Creare i mapping per le variabili categoriali in base all'ordine di importanza logico

ordinal_mappings = {
    "ExterQual": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "GarageFinish": {"Unf": 0, "RFn": 1, "Fin": 2},
    "KitchenQual": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "BsmtQual": {"NoBsmt": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "FireplaceQu": {"NoFireplace": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageCond": {"NoGarage": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageQual": {"NoGarage": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "CentralAir": {"N": 0, "Y": 1},  # Semplice binario
    "BsmtExposure": {"NoBsmt": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtCond": {"NoBsmt": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
}

# Applicare il mapping alle colonne categoriali
for col, mapping in ordinal_mappings.items():
    df[col] = df[col].map(mapping)
    
df["Price_per_sqm"] = df["SalePrice"] / df["GrLivArea"]

# Creare una mappa basata sull'ordinamento del prezzo medio al metro quadro per quartiere
neighborhood_ordered_mapping = {
    neigh: idx for idx, neigh in enumerate(df.groupby("Neighborhood")["Price_per_sqm"].mean().sort_values().index)
}

# Sostituire i nomi dei quartieri con i loro valori numerici ordinati
df["Neighborhood"] = df["Neighborhood"].map(neighborhood_ordered_mapping)

# Verificare il risultato
df["Neighborhood"].head()

# Calcolare la matrice di correlazione
correlation_matrix = df[top_features + ['SalePrice']].corr()

# Visualizzare la matrice di correlazione
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matrice di Correlazione delle 20 Feature più Importanti")
plt.show()

# Identificare le feature numeriche tra le 9 restanti dalla matrice di correlazione
top_numerical_features = ["GarageCars", "TotalBsmtSF", "1stFlrSF", "GrLivArea", 
                          "YearRemodAdd", "LotArea", "2ndFlrSF", "BsmtFinSF1", "GarageCond"]

# Impostare un limite massimo basato sui valori osservati nel dataset
feature_max_values = {} #Crea un dizionario vuoto {} e lo popola con coppie chiave-valore (feature: max_value)
for feature in top_numerical_features: #feature è la chiave del dizionario (Nome della colonna). #df[feature].max() è il valore massimo osservato per quella colonna.
    if feature in df.columns and not pd.api.types.is_categorical_dtype(df[feature]): #Verifica se la feature esiste nel DataFrame. if feature in df.columns. Verifica che la feature sia numerica 
        feature_max_values[feature] = df[feature].max() #assegna il valore massimo osservato nel dataset per una determinata feature numerica e lo salva nel dizionario feature_max_values.

feature_max_values["OverallQual"] = 10  # Limite massimo per OverallQual
feature_max_values["Neighborhood"] = len(neighborhood_ordered_mapping) - 1  # Limite massimo per Neighborhood

# Dizionario per memorizzare le preferenze
preferences = {}

# Chiedere input all'utente per ogni feature
print("Inserisci le tue preferenze per le seguenti caratteristiche:")
for feature in top_features: #Scorre tutte le feature contenute in top_features, che sono le 20 più importanti selezionate dal modello XGBoost
    while True: #while True per continuare a chiedere input finché l'utente non inserisce un valore valido.
        try:
            max_value = feature_max_values.get(feature, None) #Cerca il valore massimo nella tabella dei limiti numerici
            ordinal_max_value = max(ordinal_mappings[feature].values()) if feature in ordinal_mappings else None #trova il massimo valore nel mapping categoriale.
            overall_max_value = ordinal_max_value if ordinal_max_value is not None else max_value #Usa il massimo tra quelli disponibili.
            value = int(input(f"{feature} (Max: {overall_max_value if overall_max_value else 'Nessun limite'}): ")) #chiede all'utente di inserire un valore per la feature corrente.

            if overall_max_value and value > overall_max_value: #Controlla se il valore supera il massimo
                print(f"Il valore inserito supera il massimo consentito ({overall_max_value}). Riprova.") #continue forza la ripetizione del ciclo, evitando di salvare valori non validi.
                continue
            preferences[feature] = value #Se il valore è valido, lo memorizza nel dizionario preferences.
            break
        except ValueError: #Intercetta errori quando input() non può essere convertito in int.
            print("Inserisci un valore numerico valido.")

# Stampare le preferenze inserite
print("Preferenze salvate:")
for feature, value in preferences.items(): #Itera sul dizionario preferences
    print(f"{feature}: {value}")

# Nuove feature selezionate dall'utente
top_features_updated = [
    "OverallQual", "GarageCars", "ExterQual", "GarageFinish", "KitchenQual", 
    "GrLivArea", "TotalBsmtSF", "BsmtFinSF1", "1stFlrSF", "BsmtQual", 
    "FireplaceQu", "Neighborhood", "GarageCond", "2ndFlrSF", "GarageQual", 
    "CentralAir", "YearRemodAdd", "BsmtExposure", "LotArea", "BsmtCond"
]

# Definire gli iperparametri ottimizzati
optimized_params = {
    'n_estimators': 50,   # Numero ridotto di alberi per velocizzare il training
    'learning_rate': 0.1,
    'max_depth': 3,       # Alberi meno profondi per ridurre il tempo di addestramento
    'tree_method': "hist",
    'lambda': 1,
    'alpha': 0.5
}


# Creare un nuovo DataFrame con solo le feature selezionate
X = df[top_features_updated]
y = df['SalePrice']

# Convertire le variabili categoriche in numeriche
X = pd.get_dummies(X, drop_first=True)

# Suddividere il dataset in training (50%) e test (50%) per ridurre i tempi di calcolo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Addestrare il modello XGBoost con le nuove impostazioni ottimizzate
optimized_xgb = xgb.XGBRegressor(**optimized_params, objective="reg:squarederror", random_state=42)
optimized_xgb.fit(X_train, y_train)

# Fare previsioni sui dati di test
y_pred_updated = optimized_xgb.predict(X_test)

# Calcolare le metriche di valutazione
rmse_updated = np.sqrt(mean_squared_error(y_test, y_pred_updated))
r2_updated = r2_score(y_test, y_pred_updated)

# Creare un DataFrame con i valori reali e predetti (aggiunto per evitare l'errore)
predictions_updated_df = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred_updated
})

# Mostrare le metriche aggiornate
rmse_updated, r2_updated

# Stampare le prime righe delle previsioni
print("Prime previsioni sui prezzi delle case:")
print(predictions_updated_df.head())

# Stampare le metriche aggiornate
print("\nMetriche del modello:")
print(f"RMSE: {rmse_updated:.2f}")
print(f"R²: {r2_updated:.4f}")






