import torch
import torch.nn as nn
import torch.nn.functional as F

class IDSNet(nn.Module):
    """
    Arquitectura de Red Neuronal Profunda (DNN) para el Sistema de Detección de Intrusos.
    Diseñada para procesar las características normalizadas del dataset UNSW-NB15.
    """
    def __init__(self, input_dim, num_classes=2):
        super(IDSNet, self).__init__()
        
        # La dimensión de entrada (input_dim) dependerá de cuántas características 
        # queden después del preprocesamiento (aprox. 42-45 dependiendo del encoding).
        
        # Capa de entrada a primera capa oculta
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128) # Normalización por lotes para estabilizar el entrenamiento
        self.drop1 = nn.Dropout(0.3)
        
        # Segunda capa oculta
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.3)
        
        # Tercera capa oculta
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.drop3 = nn.Dropout(0.2)
        
        # Capa de salida
        # num_classes=2 para clasificación binaria (Normal vs Ataque)
        # o superior si decides clasificar tipos específicos de ataques.
        self.out = nn.Linear(32, num_classes)

    def forward(self, x):
        # Propagación hacia adelante con activación ReLU
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.drop3(x)
        
        # Retornamos los logits en bruto. 
        # La función de pérdida (ej. CrossEntropyLoss) se encargará del Softmax internamente.
        x = self.out(x)
        return x