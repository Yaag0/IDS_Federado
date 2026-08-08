import torch
import torch.nn as nn
import torch.optim as optim
import copy
import sys
import os
import pickle
import pandas as pd

# Asegurar que el directorio src sea reconocible como paquete
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.preprocessing import preprocess_unsw_nb15, create_non_iid_splits
from src.models.ids_net import IDSNet
from src.models.federated import FederatedAggregator
from src.security.crypto import PQCAESTunnel

# Configuración de hiperparámetros
ROUNDS = 5
NUM_CLIENTS = 3
LOCAL_EPOCHS = 2
LEARNING_RATE = 0.01
DATASET_PATH = "data/raw/UNSW_NB15_training-set.csv"

def train_local_client(client_id, model, data, labels, criterion, tunnel, server_pub_key):
    """
    Simula el entrenamiento local de un nodo cliente y el envío seguro de sus pesos.
    """
    print(f"\n--- [Cliente {client_id}] Iniciando entrenamiento local ---")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Convertir datos a tensores
    inputs = torch.FloatTensor(data)
    targets = torch.LongTensor(labels)
    
    for epoch in range(LOCAL_EPOCHS):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
    final_loss = loss.item()
    print(f"[Cliente {client_id}] Entrenamiento finalizado. Loss final: {final_loss:.4f}")
    
    # Extraer los pesos actualizados
    local_weights = model.state_dict()
    
    # ---------------------------------------------------------
    # FASE DE SEGURIDAD POST-CUÁNTICA (CLIENTE)
    # ---------------------------------------------------------
    print(f"[Cliente {client_id}] Encapsulando clave simétrica (ML-KEM / Kyber-768)...")
    ciphertext, shared_secret = tunnel.client_encapsulate(server_pub_key)
    
    # Serializar los pesos para enviarlos
    plaintext_weights = pickle.dumps(local_weights)
    
    print(f"[Cliente {client_id}] Cifrando gradientes (AES-256-GCM)...")
    encrypted_payload = tunnel.encrypt_payload(shared_secret, plaintext_weights)
    
    return ciphertext, encrypted_payload, final_loss


def guardar_metricas_csv(historial_loss):
    """Guarda las métricas en un archivo CSV para la lectura del dashboard en tiempo real."""
    df_metrics = pd.DataFrame(historial_loss)
    df_metrics.to_csv("metrics.csv", index=False)


def main():
    print("==========================================================")
    print(" INICIANDO QUANTUM SECURITY IDS FEDERADO ")
    print("==========================================================\n")

    # 1. Preparar datos
    try:
        (X_train, y_train), (X_test, y_test) = preprocess_unsw_nb15(DATASET_PATH)
        client_splits = create_non_iid_splits(X_train, y_train, num_clients=NUM_CLIENTS)
    except FileNotFoundError:
        print(f"[!] Error: No se encontró el dataset en {DATASET_PATH}.")
        print("Por favor, asegúrate de colocar el CSV en la carpeta 'data/'.")
        return

    input_dim = X_train.shape[1]

    # 2. Inicializar Modelo Global y Agregador
    global_model = IDSNet(input_dim=input_dim, num_classes=2)
    aggregator = FederatedAggregator(defense_mechanism='trimmed_mean', trim_ratio=0.2)
    criterion = nn.CrossEntropyLoss()
    tunnel = PQCAESTunnel(kem_alg="Kyber768")

    historial_loss = []

    for round_num in range(1, ROUNDS + 1):
        print(f"\n==================== RONDA FEDERADA {round_num}/{ROUNDS} ====================")
        
        # 3. Generar par de llaves PQC del Servidor Central para esta ronda
        print("[Servidor] Generando llaves PQC (ML-KEM)...")
        server_pub_key, server_priv_key = tunnel.generate_server_keypair()
        
        client_weights_list = []
        round_losses = {}
        
        # 4. Flujo de los Clientes
        for client_id in range(NUM_CLIENTS):
            client_data, client_labels = client_splits[client_id]
            
            # Cada cliente parte del modelo global actual
            local_model = copy.deepcopy(global_model)
            
            # Entrenamiento local y transmisión segura
            ciphertext, encrypted_payload, client_loss = train_local_client(
                client_id, local_model, client_data, client_labels, 
                criterion, tunnel, server_pub_key
            )
            
            round_losses[f"Cliente_{client_id}"] = client_loss
            
            # ---------------------------------------------------------
            # FASE DE SEGURIDAD POST-CUÁNTICA (SERVIDOR)
            # ---------------------------------------------------------
            shared_secret = tunnel.server_decapsulate(server_priv_key, ciphertext)
            decrypted_data = tunnel.decrypt_payload(shared_secret, encrypted_payload)
            received_weights = pickle.loads(decrypted_data)
            
            client_weights_list.append(received_weights)
            print(f"[Servidor] Gradientes del Cliente {client_id} recibidos y descifrados con éxito.")

        # Registrar métricas de la ronda actual
        historial_loss.append({
            "Ronda": round_num,
            "Cliente_0": round_losses.get("Cliente_0", 0),
            "Cliente_1": round_losses.get("Cliente_1", 0),
            "Cliente_2": round_losses.get("Cliente_2", 0)
        })
        guardar_metricas_csv(historial_loss)

        # 5. Agregación Robusta en el Servidor
        print("\n[Servidor] Agregando pesos usando defensa: Trimmed Mean...")
        new_global_weights = aggregator.aggregate(client_weights_list)
        global_model.load_state_dict(new_global_weights)
        print(f"[Servidor] Modelo global actualizado para la ronda {round_num}.")

    print("\n==========================================================")
    print(" ENTRENAMIENTO FEDERADO COMPLETADO ")
    print("==========================================================")

if __name__ == "__main__":
    main()