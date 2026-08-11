import pandas as pd
import matplotlib.pyplot as plt
import os

def graficar_comportamiento_clientes(csv_path="results/metrics.csv"):
    if not os.path.exists(csv_path):
        print(f"Archivo {csv_path} no encontrado.")
        return
        
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    
    # Graficar atacantes (rojo y naranja)
    plt.plot(df['Ronda'], df['Cliente_0'], label='Cliente 0 (Ataque: Ruido)', color='red', linestyle='--')
    plt.plot(df['Ronda'], df['Cliente_1'], label='Cliente 1 (Ataque: Escalado)', color='orange', linestyle='--')
    
    # Graficar clientes honestos (tonos de azul/verde)
    plt.plot(df['Ronda'], df['Cliente_2'], label='Cliente 2 (Honesto)', color='blue')
    plt.plot(df['Ronda'], df['Cliente_3'], label='Cliente 3 (Honesto)', color='green')
    plt.plot(df['Ronda'], df['Cliente_4'], label='Cliente 4 (Honesto)', color='teal')
    
    plt.title("Comportamiento de la Pérdida (Loss) por Cliente durante el Entrenamiento")
    plt.xlabel("Rondas de Federación")
    plt.ylabel("Pérdida (CrossEntropyLoss)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("results/ataques_vs_honestos.png", dpi=300)
    print("Gráfica guardada en results/ataques_vs_honestos.png")

if __name__ == "__main__":
    graficar_comportamiento_clientes()