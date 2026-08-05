import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

def evaluate_model(model: torch.nn.Module, dataloader, device: torch.device):
    """
    Evalúa el modelo de PyTorch utilizando un DataLoader y devuelve las 
    etiquetas reales, las predicciones y las probabilidades.
    """
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []
    
    with torch.no_grad():
        for Xb, yb in dataloader:
            Xb = Xb.to(device)
            outputs = model(Xb)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            y_true.extend(yb.numpy())
            y_pred.extend(preds)
            y_probs.extend(probs)
            
    return np.array(y_true), np.array(y_pred), np.array(y_probs)

def print_metrics(y_true, y_pred, strategy_name: str = "Modelo"):
    """Imprime por consola las métricas principales de clasificación."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n--- Resultados para {strategy_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

def save_confusion_matrix(y_true, y_pred, save_path: str, title: str = "Matriz de Confusión"):
    """Genera y guarda la matriz de confusión en formato PNG."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_roc_curve(y_true, y_probs, save_path: str, title: str = "Curva ROC"):
    """Genera y guarda la Curva ROC en formato PNG."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()