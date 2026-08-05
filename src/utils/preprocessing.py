import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch

def load_and_preprocess_data(data_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Carga el dataset UNSW-NB15, maneja valores nulos, codifica variables categóricas,
    normaliza las características y devuelve los tensores listos para PyTorch.
    """
    file_path = os.path.join(data_path, "UNSW_NB15_training-set.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el dataset en: {file_path}. Por favor, colócalo en la carpeta data/raw/")

    print(f"Cargando dataset desde {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Dataset cargado exitosamente. Dimensiones iniciales: {df.shape}")

    # Copia de seguridad y validación de etiqueta
    data = df.copy()
    if 'label' not in data.columns:
        raise ValueError("El dataset no contiene la columna obligatoria 'label'.")
    
    data['label'] = data['label'].astype(int)

    # Codificación de columnas categóricas de forma segura
    cat_cols = data.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

    # Separación de características (X) y etiquetas (y)
    X = data.drop(columns=['label'])
    y = data['label'].astype(int)

    # Escalado de características (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # División estratificada de los datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Conversión a tensores de PyTorch (optimizados para float32)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    print(f"Datos preprocesados -> Train: {X_train_t.shape}, Test: {X_test_t.shape}")
    
    return X_train_t, y_train_t, X_test_t, y_test_t, X.columns.tolist()