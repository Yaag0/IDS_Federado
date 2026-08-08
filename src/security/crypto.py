import oqs
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class PQCAESTunnel:
    """
    Túnel Híbrido PQC-AES.
    Utiliza ML-KEM (Kyber) para el intercambio seguro de llaves post-cuánticas
    y AES-256-GCM para el cifrado de alto rendimiento de los gradientes.
    """
    def __init__(self, kem_alg="Kyber768"):
        self.kem_alg = kem_alg
        # Mantenemos el objeto KEM del servidor instanciado para conservar 
        # intactas las referencias de memoria C (_ctypes) entre las fases.
        self.server_kem = oqs.KeyEncapsulation(self.kem_alg)

    def generate_server_keypair(self):
        """Genera el par de llaves post-cuánticas del servidor."""
        pub_key = self.server_kem.generate_keypair()
        priv_key = self.server_kem.export_secret_key()
        return pub_key, priv_key

    def client_encapsulate(self, server_pub_key):
        """El cliente usa la llave pública del servidor para encapsular un secreto."""
        with oqs.KeyEncapsulation(self.kem_alg) as client_kem:
            ciphertext, shared_secret = client_kem.encap_secret(server_pub_key)
        return ciphertext, shared_secret

    def server_decapsulate(self, server_priv_key, ciphertext):
        """El servidor desencapsula el secreto usando su estado interno seguro."""
        # Al usar self.server_kem, evitamos inyectar 'bytes' crudos
        # y permitimos que la librería en C maneje su propia decapsulación.
        shared_secret = self.server_kem.decap_secret(ciphertext)
        return shared_secret

    def encrypt_payload(self, shared_secret, plaintext_data):
        """Cifra los pesos del modelo usando AES-256-GCM con el secreto PQC."""
        aesgcm = AESGCM(shared_secret[:32])
        nonce = os.urandom(12) 
        ciphertext = aesgcm.encrypt(nonce, plaintext_data, None)
        return nonce + ciphertext

    def decrypt_payload(self, shared_secret, encrypted_payload):
        """Descifra los pesos recibidos en el servidor."""
        aesgcm = AESGCM(shared_secret[:32])
        nonce = encrypted_payload[:12]
        ciphertext = encrypted_payload[12:]
        plaintext_data = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext_data

    def __del__(self):
        """Libera la memoria de C de forma segura al destruir el objeto."""
        if hasattr(self, 'server_kem'):
            self.server_kem.free()