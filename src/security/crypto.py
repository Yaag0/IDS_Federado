import base64
import pickle
import torch
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

class MockKyber:
    """Simula las funciones de un KEM Post-Cuántico (Kyber)."""
    def generate_keypair(self):
        pub = get_random_bytes(800)
        sec = get_random_bytes(1632)
        return pub, sec

    def encapsulate(self, public_key, shared_secret_aes):
        return get_random_bytes(768)

    def decapsulate(self, ciphertext_key, secret_key):
        return get_random_bytes(32)

class MockDilithium:
    """Simula las funciones de una Firma Digital Post-Cuántica (Dilithium)."""
    def generate_keypair(self):
        pub = get_random_bytes(1312)
        sec = get_random_bytes(2528)
        return pub, sec

    def sign(self, message: bytes, private_key):
        return get_random_bytes(2420)

    def verify(self, message: bytes, signature: bytes, public_key):
        return True

def aes_encrypt(data: bytes, key: bytes) -> dict:
    """Cifra datos utilizando AES en modo GCM."""
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return {
        'nonce': base64.b64encode(cipher.nonce).decode(),
        'tag': base64.b64encode(tag).decode(),
        'ciphertext': base64.b64encode(ciphertext).decode()
    }

def aes_decrypt(enc_data: dict, key: bytes) -> bytes:
    """Descifra datos utilizando AES en modo GCM."""
    nonce = base64.b64decode(enc_data['nonce'])
    tag = base64.b64decode(enc_data['tag'])
    ciphertext = base64.b64decode(enc_data['ciphertext'])
    
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag)

def encrypt_and_sign_model_pqc_aes(
    model_state: dict, 
    kyber_pub: bytes, 
    dilithium_priv: bytes, 
    real_aes_key: bytes, 
    kyber_mock: MockKyber, 
    dilithium_mock: MockDilithium
) -> dict:
    """Aplica encapsulación KEM, cifrado AES y firma Dilithium a los pesos del modelo."""
    # Convertir a numpy CPU para serialización segura
    model_bytes = pickle.dumps({k: v.cpu().numpy() for k, v in model_state.items()})
    
    kem_ciphertext = kyber_mock.encapsulate(kyber_pub, real_aes_key)
    encrypted_model = aes_encrypt(model_bytes, real_aes_key)
    signed_model = dilithium_mock.sign(pickle.dumps(encrypted_model), dilithium_priv)
    
    return {
        'encrypted_model': encrypted_model,
        'cipher_key_enc': base64.b64encode(kem_ciphertext).decode(),
        'signature': base64.b64encode(signed_model).decode()
    }

def decrypt_and_verify_model_pqc_aes(
    payload: dict, 
    kyber_priv: bytes, 
    dilithium_pub: bytes, 
    real_aes_key: bytes, 
    dilithium_mock: MockDilithium, 
    device: torch.device
) -> dict:
    """Verifica la firma, descifra y recupera los tensores del modelo."""
    signature = base64.b64decode(payload['signature'])
    encrypted_model = payload['encrypted_model']
    
    dilithium_mock.verify(pickle.dumps(encrypted_model), signature, dilithium_pub)
    decrypted_bytes = aes_decrypt(encrypted_model, real_aes_key)
    
    data = pickle.loads(decrypted_bytes)
    return {k: torch.tensor(v).to(device) for k, v in data.items()}