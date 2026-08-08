import torch
import copy

class FederatedAggregator:
    """
    Módulo para la agregación robusta de pesos en el Aprendizaje Federado.
    Implementa defensas contra Model Poisoning.
    """
    def __init__(self, defense_mechanism='trimmed_mean', trim_ratio=0.2):
        self.defense_mechanism = defense_mechanism
        self.trim_ratio = trim_ratio

    def aggregate(self, client_weights):
        """
        Recibe una lista de state_dicts (pesos de los clientes) y retorna el state_dict agregado.
        """
        if not client_weights:
            raise ValueError("La lista de pesos de los clientes está vacía.")

        if self.defense_mechanism == 'trimmed_mean':
            return self._trimmed_mean(client_weights)
        elif self.defense_mechanism == 'median':
            return self._median(client_weights)
        else:
            return self._fed_avg(client_weights)

    def _fed_avg(self, client_weights):
        """Federated Averaging estándar (sin defensa)."""
        global_weights = copy.deepcopy(client_weights[0])
        num_clients = len(client_weights)

        for key in global_weights.keys():
            stacked_weights = torch.stack([client_weights[i][key] for i in range(num_clients)])
            
            # Verificación de tipo de dato para evitar errores con tensores Long (ej. num_batches_tracked)
            if not stacked_weights.is_floating_point():
                global_weights[key] = torch.mean(stacked_weights.float(), dim=0).to(stacked_weights.dtype)
            else:
                global_weights[key] = torch.mean(stacked_weights, dim=0)
                
        return global_weights

    def _trimmed_mean(self, client_weights):
        """
        Media Recortada: Elimina los valores extremos antes de promediar.
        """
        global_weights = copy.deepcopy(client_weights[0])
        num_clients = len(client_weights)
        trim_count = int(num_clients * self.trim_ratio)

        if num_clients <= 2 * trim_count:
            return self._fed_avg(client_weights)

        for key in global_weights.keys():
            stacked_weights = torch.stack([client_weights[i][key] for i in range(num_clients)])
            sorted_weights, _ = torch.sort(stacked_weights, dim=0)
            trimmed_weights = sorted_weights[trim_count : num_clients - trim_count]
            
            # Verificación de tipo de dato para PyTorch
            if not trimmed_weights.is_floating_point():
                global_weights[key] = torch.mean(trimmed_weights.float(), dim=0).to(trimmed_weights.dtype)
            else:
                global_weights[key] = torch.mean(trimmed_weights, dim=0)
            
        return global_weights

    def _median(self, client_weights):
        """
        Mediana Robusta: Toma la mediana de los pesos.
        """
        global_weights = copy.deepcopy(client_weights[0])
        num_clients = len(client_weights)

        for key in global_weights.keys():
            stacked_weights = torch.stack([client_weights[i][key] for i in range(num_clients)])
            median_weights, _ = torch.median(stacked_weights, dim=0)
            global_weights[key] = median_weights
            
        return global_weights