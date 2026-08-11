import torch

def apply_model_poisoning(weights, attack_type="noise", noise_std=0.5, scaling_factor=100.0):
    poisoned_weights = {}
    
    for name, param in weights.items():
        # Los parámetros como 'num_batches_tracked' en BatchNorm son enteros (Long).
        # No se les puede aplicar ruido gaussiano ni escalado decimal, así que los saltamos.
        if not param.is_floating_point():
            poisoned_weights[name] = param.clone()
            continue

        if attack_type == "noise":
            noise = torch.randn_like(param) * noise_std
            poisoned_weights[name] = param + noise
            
        elif attack_type == "scaling":
            poisoned_weights[name] = param * scaling_factor
            
        else:
            poisoned_weights[name] = param.clone()
            
    return poisoned_weights