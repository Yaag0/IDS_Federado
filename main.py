import os
import torch
from torch.utils.data import TensorDataset, DataLoader

# Importaciones de los módulos locales
from src.utils.preprocessing import load_and_preprocess_data
from src.models.ids_net import IDSNetBinario
from src.models.federated import train_local, fedavg, krum_defense
from src.security.crypto import (
    MockKyber, MockDilithium, 
    encrypt_and_sign_model_pqc_aes, 
    decrypt_and_verify_model_pqc_aes
)
from src.utils.metrics import evaluate_model, print_metrics, save_confusion_matrix, save_roc_curve
from Crypto.Random import get_random_bytes

def main():
    # 1. Configuración inicial y directorios
    print("Iniciando orquestación del sistema...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo de entrenamiento: {device}")
    
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    
    # 2. Carga y preprocesamiento de datos
    # NOTA: Asegúrate de tener el archivo UNSW_NB15_training-set.csv en data/raw/
    try:
        X_train, y_train, X_test, y_test, features = load_and_preprocess_data(data_path='data/raw')
    except FileNotFoundError as e:
        print(f"Error crítico: {e}")
        return

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256, shuffle=False)

    # 3. Inicialización del modelo global
    input_dim = X_train.shape[1]
    global_model = IDSNetBinario(input_dim).to(device)
    
    # 4. Simulación de Aprendizaje Federado
    print("\n--- Iniciando Aprendizaje Federado (Simulación 3 clientes) ---")
    client_weights = []
    for client_id in range(3):
        print(f"Entrenando Cliente {client_id + 1}...")
        pesos = train_local(global_model, train_loader, device, epochs=3)
        client_weights.append(pesos)
        
    print("\nAgregando pesos con defensa Krum...")
    global_weights = krum_defense(client_weights)
    global_model.load_state_dict(global_weights)
    
    # 5. Seguridad PQC (Cifrado Híbrido)
    print("\n--- Asegurando modelo con PQC y AES ---")
    kyber = MockKyber()
    dilithium = MockDilithium()
    
    kyber_pub, _ = kyber.generate_keypair()
    dilithium_pub, dilithium_priv = dilithium.generate_keypair()
    real_aes_key = get_random_bytes(32)
    
    # Simular servidor enviando modelo cifrado
    payload = encrypt_and_sign_model_pqc_aes(
        global_weights, kyber_pub, dilithium_priv, real_aes_key, kyber, dilithium
    )
    print("Modelo cifrado y firmado exitosamente.")
    
    # Simular cliente recibiendo y descifrando modelo
    decrypted_weights = decrypt_and_verify_model_pqc_aes(
        payload, b'dummy_kyber_priv', dilithium_pub, real_aes_key, dilithium, device
    )
    global_model.load_state_dict(decrypted_weights)
    print("Modelo descifrado y verificado exitosamente.")
    
    # 6. Evaluación
    print("\n--- Evaluando Modelo Global ---")
    y_true, y_pred, y_probs = evaluate_model(global_model, test_loader, device)
    
    metrics = print_metrics(y_true, y_pred, strategy_name="Modelo Global Krum + PQC")
    
    # Generar y guardar figuras
    save_confusion_matrix(y_true, y_pred, save_path='reports/figures/confusion_matrix.png')
    save_roc_curve(y_true, y_probs, save_path='reports/figures/roc_curve.png')
    print("\nGráficas guardadas en reports/figures/")
    print("¡Ejecución finalizada con éxito!")

if __name__ == "__main__":
    main()