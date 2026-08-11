import pandas as pd
import matplotlib.pyplot as plt
import os

def graficar_latencias(csv_path="results/latencias_agregacion.csv"):
    if not os.path.exists(csv_path):
        print(f"Archivo {csv_path} no encontrado.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Calcular promedios de las 5 rondas
    avg_pqc = df['t_simulado_pqc'].mean()
    avg_he = df['t_simulado_he'].mean()
    
    plt.figure(figsize=(8, 6))
    barras = plt.bar(['PQC (Kyber768)', 'Cifrado Homomórfico (Simulado)'], 
                     [avg_pqc, avg_he], 
                     color=['#2ca02c', '#d62728'])
    
    plt.title('Costo Computacional: PQC vs Cifrado Homomórfico')
    plt.ylabel('Tiempo Promedio por Ronda (Milisegundos)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Añadir los valores encima de las barras
    for barra in barras:
        yval = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2, yval + 0.5, 
                 f'{yval:.1f} ms', ha='center', va='bottom', fontweight='bold')
                 
    plt.savefig("results/comparativa_latencias.png", dpi=300)
    print("Gráfica guardada en results/comparativa_latencias.png")

if __name__ == "__main__":
    graficar_latencias()