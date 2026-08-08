import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_unsw_nb15(filepath):
    print("[*] Cargando el dataset UNSW-NB15...")
    df = pd.read_csv(filepath)

    # Asegurar que los nombres de las columnas no tengan espacios en blanco
    df.columns = df.columns.str.strip()

    # Eliminar la columna de ID y la subcategoría de ataque (attack_cat) 
    # para evitar fuga de datos (Data Leakage) en nuestra clasificación binaria
    cols_to_drop = [col for col in ['id', 'attack_cat', 'Attack_cat'] if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Identificar la columna objetivo (normal vs ataque)
    target_col = 'label' if 'label' in df.columns else 'Label'
    
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    # ---------------------------------------------------------
    # CORRECCIÓN: CODIFICACIÓN DE VARIABLES CATEGÓRICAS (TEXTO)
    # ---------------------------------------------------------
    print("[*] Transformando variables de texto a valores numéricos...")
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Convertimos texto como 'udp', 'tcp', 'dns' en números enteros como 0, 1, 2...
        X[col] = le.fit_transform(X[col].astype(str))

    # ---------------------------------------------------------
    # ESCALADO MIN-MAX
    # ---------------------------------------------------------
    print("[*] Aplicando Min-Max Scaling [0, 1]...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Separar en conjunto de entrenamiento y validación
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return (X_train, y_train), (X_test, y_test)


def create_non_iid_splits(X, y, num_clients):
    """
    Divide los datos de forma Non-IID (No independientes e idénticamente distribuidos)
    para simular un entorno realista de Aprendizaje Federado.
    """
    print(f"[*] Creando particiones de datos para {num_clients} clientes...")
    
    # Para este PoC, hacemos una distribución heterogénea aleatorizada simple
    splits = []
    data_len = len(X)
    indices = np.random.permutation(data_len)
    
    chunk_size = data_len // num_clients
    
    for i in range(num_clients):
        start = i * chunk_size
        # El último cliente se lleva el residuo de datos
        end = start + chunk_size if i != num_clients - 1 else data_len
        
        client_idx = indices[start:end]
        splits.append((X[client_idx], y[client_idx]))
        
    return splits