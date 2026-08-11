import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

def evaluate_and_save_metrics(model, dataloader, device, scenario_name, results_dir="results"):
    """Evalúa el modelo y exporta métricas, matriz de confusión y curvas ROC."""
    os.makedirs(results_dir, exist_ok=True)
    model.eval()
    
    y_true_list, probs_list, preds_list = [], [], []
    
    with torch.no_grad():
        for Xb, yb in dataloader:
            Xb = Xb.to(device)
            out = model(Xb)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(out, dim=1).cpu().numpy()
            
            probs_list.extend(probs)
            preds_list.extend(preds)
            y_true_list.extend(yb.cpu().numpy().ravel().astype(int))
            
    # 1. Calcular métricas clave
    acc = accuracy_score(y_true_list, preds_list)
    prec = precision_score(y_true_list, preds_list, zero_division=0)
    rec = recall_score(y_true_list, preds_list, zero_division=0)
    f1 = f1_score(y_true_list, preds_list, zero_division=0)
    
    metrics_dict = {
        'escenario': scenario_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }
    
    # Guardar en CSV (Append)
    csv_path = os.path.join(results_dir, "metricas_completas.csv")
    df = pd.DataFrame([metrics_dict])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

    # 2. Exportar Matriz de Confusión
    cm = confusion_matrix(y_true_list, preds_list)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Ataque'], yticklabels=['Normal', 'Ataque'])
    plt.title(f"Matriz Confusión - {scenario_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"cm_{scenario_name}.png"), dpi=300)
    plt.close()

    # 3. Exportar Curva ROC
    fpr, tpr, _ = roc_curve(y_true_list, probs_list)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC - {scenario_name}')  # <-- CORREGIDO AQUÍ (plt.title en lugar de plt.setTitle)
    plt.legend()
    plt.savefig(os.path.join(results_dir, f'roc_{scenario_name}.png'), dpi=300)
    plt.close()

    print(f"✅ Artefactos generados para {scenario_name} en la carpeta '{results_dir}/'")
    return metrics_dict