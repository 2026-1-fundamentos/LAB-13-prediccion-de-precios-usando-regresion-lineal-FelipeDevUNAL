#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import pandas as pd
import json
import pickle
import gzip
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error


def preprocesar(df):
    df = df.copy()
    df["Age"] = 2021 - df["Year"]
    df = df.drop(columns=["Year", "Car_Name"])
    return df


def crear_modelo():
    cat_cols = ["Fuel_Type", "Selling_type", "Transmission"]
    num_cols = ["Selling_Price", "Driven_kms", "Owner", "Age"]
    
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(), cat_cols),
        ("num", MinMaxScaler(), num_cols)
    ])
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("selector", SelectKBest(f_regression)),
        ("regressor", LinearRegression())
    ])
    
    params = {
        "selector__k": list(range(1, 15)),
        "regressor__fit_intercept": [True, False],
        "regressor__positive": [True, False]
    }
    
    grid = GridSearchCV(
        pipeline,
        params,
        cv=10,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    
    return grid


def calcular_metricas(nombre, y_real, y_pred):
    return {
        "type": "metrics",
        "dataset": nombre,
        "r2": float(r2_score(y_real, y_pred)),
        "mse": float(mean_squared_error(y_real, y_pred)),
        "mad": float(median_absolute_error(y_real, y_pred))
    }


if __name__ == "__main__":
    
    train_df = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
    test_df = pd.read_csv("files/input/test_data.csv.zip", compression="zip")
    
    train_df = preprocesar(train_df)
    test_df = preprocesar(test_df)
    
    X_train = train_df.drop(columns=["Present_Price"])
    y_train = train_df["Present_Price"]
    X_test = test_df.drop(columns=["Present_Price"])
    y_test = test_df["Present_Price"]
    
    modelo = crear_modelo()
    modelo.fit(X_train, y_train)
    
    Path("files/models").mkdir(parents=True, exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(modelo, f)
    
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    
    metricas_train = calcular_metricas("train", y_train, y_pred_train)
    metricas_test = calcular_metricas("test", y_test, y_pred_test)
    
    Path("files/output").mkdir(parents=True, exist_ok=True)
    with open("files/output/metrics.json", "w") as f:   
        f.write(json.dumps(metricas_train) + "\n")
        f.write(json.dumps(metricas_test) + "\n")