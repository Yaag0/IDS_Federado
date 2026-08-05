import copy
import torch
import numpy as np

def train_local(model: torch.nn.Module, dataloader, device: torch.device, epochs: int = 2, lr: float = 1e-3) -> dict:
    """
    Entrena localmente una copia del modelo global en los datos de un cliente.
    """
    local_model = copy.deepcopy(model).to(device)
    optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    local_model.train()
    for _ in range(epochs):
        for Xb, yb in dataloader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(local_model(Xb), yb)
            loss.backward()
            optimizer.step()
            
    return local_model.state_dict()

def fedavg(pesos_clientes: list) -> dict:
    """Agregación clásica FedAvg (Promedio simple de los pesos)."""
    agg = copy.deepcopy(pesos_clientes[0])
    for k in agg.keys():
        for i in range(1, len(pesos_clientes)):
            agg[k] = agg[k] + pesos_clientes[i][k]
        agg[k] = agg[k] / len(pesos_clientes)
    return agg

def median_defense(pesos_clientes: list) -> dict:
    """Defensa robusta mediante Mediana (resistente a envenenamiento de datos)."""
    with torch.no_grad():
        agregados = {}
        for k in pesos_clientes[0].keys():
            stacked = torch.stack([p[k].float() for p in pesos_clientes])
            agregados[k] = torch.median(stacked, dim=0).values
        return agregados

def trimmed_defense(pesos_clientes: list, trim_ratio: float = 0.1) -> dict:
    """Defensa robusta mediante Media Recortada (descarta los extremos)."""
    with torch.no_grad():
        agregados = {}
        for k in pesos_clientes[0].keys():
            stacked = torch.stack([p[k].float() for p in pesos_clientes])
            lower = int(trim_ratio * len(stacked))
            upper = len(stacked) - lower
            
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[lower:upper]
            agregados[k] = trimmed.mean(dim=0)
        return agregados

def krum_defense(pesos_clientes: list) -> dict:
    """
    Defensa Krum: Selecciona el modelo del cliente más cercano a la mayoría 
    (menor distancia euclidiana general).
    """
    with torch.no_grad():
        num_clients = len(pesos_clientes)
        distancias = np.zeros((num_clients, num_clients))
        
        for i in range(num_clients):
            for j in range(num_clients):
                if i != j:
                    dist = 0
                    for k in pesos_clientes[0].keys():
                        dist += torch.norm(pesos_clientes[i][k].float() - pesos_clientes[j][k].float()).item()
                    distancias[i, j] = dist
                    
        puntajes = distancias.sum(axis=1)
        idx_mejor = np.argmin(puntajes)
        
        return copy.deepcopy(pesos_clientes[idx_mejor])