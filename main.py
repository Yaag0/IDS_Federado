import torch
import torch.nn as nn
import torch.optim as optim
import copy
import sys
import os
import pickle
import time
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.evaluation import evaluate_and_save_metrics
from src.utils.preprocessing import preprocess_unsw_nb15, create_non_iid_splits
from src.models.ids_net import IDSNet
from src.models.federated import FederatedAggregator
from src.security.crypto import PQCAESTunnel
from src.security.attacks import apply_model_poisoning

ROUNDS = 5
NUM_CLIENTS = 5  
LOCAL_EPOCHS = 2
LEARNING_RATE = 0.01
DATASET_PATH = "data/raw/UNSW_NB15_training-set.csv"

def train_local_client(client_id, model, data, labels, criterion, tunnel, server_pub_key, device, attack_type=None):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    inputs = torch.FloatTensor(data).to(device)
    targets = torch.LongTensor(labels).to(device)
    
    for _ in range(LOCAL_EPOCHS):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
    local_weights = model.state_dict()
    
    if attack_type is not None:
        local_weights = apply_model_poisoning(local_weights, attack_type=attack_type)
    
    t_inicio = time.perf_counter()
    ciphertext, shared_secret = tunnel.client_encapsulate(server_pub_key)
    plaintext_weights = pickle.dumps(local_weights)
    encrypted_payload = tunnel.encrypt_payload(shared_secret, plaintext_weights)
    t_fin = time.perf_counter()
    
    latencia_pqc_ms = (t_fin - t_inicio) * 1000.0 
    
    return ciphertext, encrypted_payload, loss.item(), latencia_pqc_ms

def guardar_metricas_csv(historial_loss):
    os.makedirs("results", exist_ok=True)
    df_metrics = pd.DataFrame(historial_loss)
    df_metrics.to_csv("results/metrics.csv", index=False)

def guardar_latencias_csv(historial_latencias):
    os.makedirs("results", exist_ok=True)
    df_lat = pd.DataFrame(historial_latencias)
    df_lat.to_csv("results/latencias_agregacion.csv", index=False)

def main():
    try:
        (X_train, y_train), (X_test, y_test) = preprocess_unsw_nb15(DATASET_PATH)
        client_splits = create_non_iid_splits(X_train, y_train, num_clients=NUM_CLIENTS)
    except FileNotFoundError:
        print(f"Error: Dataset no encontrado en {DATASET_PATH}.")
        return

    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tensor_x_test = torch.FloatTensor(X_test)
    tensor_y_test = torch.LongTensor(y_test)
    test_dataset = TensorDataset(tensor_x_test, tensor_y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    global_model = IDSNet(input_dim=input_dim, num_classes=2).to(device)
    
    aggregator = FederatedAggregator(defense_mechanism='trimmed_mean', trim_ratio=0.2)
    criterion = nn.CrossEntropyLoss()
    tunnel = PQCAESTunnel(kem_alg="Kyber768")

    historial_loss = []
    historial_latencias = []

    for round_num in range(1, ROUNDS + 1):
        print(f"\n--- Inicia Ronda {round_num}/{ROUNDS} ---")
        server_pub_key, server_priv_key = tunnel.generate_server_keypair()
        client_weights_list = []
        round_losses = {}
        round_latencias = []
        
        for client_id in range(NUM_CLIENTS):
            client_data, client_labels = client_splits[client_id]
            local_model = copy.deepcopy(global_model).to(device)
            
            attack_type = None
            if client_id == 0:
                attack_type = "noise"
            elif client_id == 1:
                attack_type = "scaling"
                
            ciphertext, encrypted_payload, client_loss, lat_ms = train_local_client(
                client_id, local_model, client_data, client_labels, 
                criterion, tunnel, server_pub_key, device, attack_type=attack_type
            )
            
            round_losses[f"Cliente_{client_id}"] = client_loss
            round_latencias.append(lat_ms)
            
            shared_secret = tunnel.server_decapsulate(server_priv_key, ciphertext)
            decrypted_data = tunnel.decrypt_payload(shared_secret, encrypted_payload)
            
            client_weights_list.append(pickle.loads(decrypted_data))

        historial_loss.append({
            "Ronda": round_num,
            "Cliente_0": round_losses.get("Cliente_0", 0),
            "Cliente_1": round_losses.get("Cliente_1", 0),
            "Cliente_2": round_losses.get("Cliente_2", 0),
            "Cliente_3": round_losses.get("Cliente_3", 0),
            "Cliente_4": round_losses.get("Cliente_4", 0)
        })
        guardar_metricas_csv(historial_loss)
        
        promedio_latencia = sum(round_latencias) / len(round_latencias)
        historial_latencias.append({
            "ronda": round_num,
            "t_actual_mediana": promedio_latencia,
            "t_simulado_pqc": promedio_latencia,
            "t_simulado_he": promedio_latencia * 18.5  
        })
        guardar_latencias_csv(historial_latencias)
        
        new_global_weights = aggregator.aggregate(client_weights_list)
        global_model.load_state_dict(new_global_weights)

    evaluate_and_save_metrics(
        model=global_model,
        dataloader=test_loader,
        device=device,
        scenario_name="Fed_Trimmed_Kyber_Final",
        results_dir="results"
    )
    print("✅ Proceso de entrenamiento y evaluación federada concluido exitosamente.")

if __name__ == "__main__":
    main()