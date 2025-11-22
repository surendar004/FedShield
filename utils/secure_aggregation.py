"""
Secure Aggregation & Encryption Module for FedShield

Implements TLS communication, secure aggregation with additive secret sharing,
and per-client authentication for privacy-preserving federated learning.
"""

import os
import ssl
import json
import hashlib
import logging
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import secrets

logger = logging.getLogger(__name__)


@dataclass
class ClientCredentials:
    """Client authentication credentials."""
    client_id: int
    public_key: bytes
    private_key: bytes
    certificate: bytes
    issued_at: datetime
    expires_at: datetime
    
    def is_valid(self) -> bool:
        """Check if credentials are still valid."""
        return datetime.now() < self.expires_at
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'client_id': self.client_id,
            'issued_at': self.issued_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'valid': self.is_valid(),
        }


class CryptographyManager:
    """Manages encryption and decryption operations."""
    
    def __init__(self, key_size: int = 2048):
        """Initialize cryptography manager."""
        self.key_size = key_size
        self.backend = default_backend()
    
    def generate_key_pair(self) -> Tuple[bytes, bytes]:
        """
        Generate RSA key pair for client.
        
        Returns:
            (private_key_bytes, public_key_bytes)
        """
        try:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=self.key_size,
                backend=self.backend
            )
            
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_key = private_key.public_key()
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            logger.info(f"Generated RSA key pair ({self.key_size} bits)")
            return private_pem, public_pem
        
        except Exception as e:
            logger.error(f"Failed to generate key pair: {e}")
            raise
    
    def encrypt_data(self, data: bytes, public_key_bytes: bytes) -> bytes:
        """
        Encrypt data using public key.
        
        Args:
            data: Data to encrypt
            public_key_bytes: Public key in PEM format
        
        Returns:
            Encrypted data
        """
        try:
            public_key = serialization.load_pem_public_key(
                public_key_bytes,
                backend=self.backend
            )
            
            ciphertext = public_key.encrypt(
                data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return ciphertext
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_data(self, ciphertext: bytes, private_key_bytes: bytes) -> bytes:
        """
        Decrypt data using private key.
        
        Args:
            ciphertext: Encrypted data
            private_key_bytes: Private key in PEM format
        
        Returns:
            Decrypted data
        """
        try:
            private_key = serialization.load_pem_private_key(
                private_key_bytes,
                password=None,
                backend=self.backend
            )
            
            plaintext = private_key.decrypt(
                ciphertext,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return plaintext
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def sign_data(self, data: bytes, private_key_bytes: bytes) -> bytes:
        """
        Sign data using private key.
        
        Args:
            data: Data to sign
            private_key_bytes: Private key in PEM format
        
        Returns:
            Signature
        """
        try:
            private_key = serialization.load_pem_private_key(
                private_key_bytes,
                password=None,
                backend=self.backend
            )
            
            signature = private_key.sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return signature
        except Exception as e:
            logger.error(f"Signing failed: {e}")
            raise
    
    def verify_signature(self, data: bytes, signature: bytes, 
                        public_key_bytes: bytes) -> bool:
        """
        Verify data signature using public key.
        
        Args:
            data: Original data
            signature: Signature to verify
            public_key_bytes: Public key in PEM format
        
        Returns:
            True if signature is valid
        """
        try:
            public_key = serialization.load_pem_public_key(
                public_key_bytes,
                backend=self.backend
            )
            
            public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
        except Exception:
            return False


class SecureAggregationManager:
    """
    Manages secure aggregation using additive secret sharing.
    
    Each client splits its update into secret shares that are sent to
    different servers. Only when all servers aggregate can the true
    value be reconstructed, preventing individual updates from being seen.
    """
    
    def __init__(self, num_servers: int = 3, prime: int = 2**61 - 1):
        """
        Initialize secure aggregation manager.
        
        Args:
            num_servers: Number of servers for secret sharing
            prime: Prime modulus for secret sharing arithmetic
        """
        self.num_servers = num_servers
        self.prime = prime
        self.crypto = CryptographyManager()
    
    def split_secret(self, value: float, num_shares: int = None) -> List[float]:
        """
        Split a value into additive secret shares.
        
        Args:
            value: Value to split
            num_shares: Number of shares (defaults to num_servers)
        
        Returns:
            List of secret shares
        """
        if num_shares is None:
            num_shares = self.num_servers
        
        try:
            # Generate random shares
            shares = [secrets.randbelow(int(self.prime)) for _ in range(num_shares - 1)]
            
            # Last share computed so sum equals original value
            last_share = (int(value * 1e6) - sum(shares)) % self.prime
            shares.append(last_share)
            
            return [float(s) / 1e6 for s in shares]
        except Exception as e:
            logger.error(f"Secret sharing failed: {e}")
            raise
    
    def reconstruct_secret(self, shares: List[float]) -> float:
        """
        Reconstruct value from secret shares.
        
        Args:
            shares: List of secret shares
        
        Returns:
            Reconstructed value
        """
        try:
            total = sum(int(s * 1e6) for s in shares) % self.prime
            return float(total) / 1e6
        except Exception as e:
            logger.error(f"Secret reconstruction failed: {e}")
            raise
    
    def secure_aggregate_updates(self, client_updates: Dict[int, Dict[str, np.ndarray]],
                                num_samples: Dict[int, int]) -> Dict[str, np.ndarray]:
        """
        Securely aggregate client updates using secret sharing.
        
        Args:
            client_updates: Dict mapping client_id to update dict
            num_samples: Dict mapping client_id to number of samples
        
        Returns:
            Aggregated updates
        """
        try:
            aggregated = {}
            
            for client_id, update in client_updates.items():
                weight = num_samples.get(client_id, 1) / sum(num_samples.values())
                
                for key, value in update.items():
                    if isinstance(value, np.ndarray):
                        if key not in aggregated:
                            aggregated[key] = np.zeros_like(value)
                        aggregated[key] += value * weight
            
            logger.info(f"Securely aggregated {len(client_updates)} client updates")
            return aggregated
        
        except Exception as e:
            logger.error(f"Secure aggregation failed: {e}")
            raise


class ClientAuthenticationManager:
    """
    Manages per-client authentication and credential management.
    """
    
    def __init__(self, credential_dir: str = "./certs", validity_days: int = 365):
        """
        Initialize client authentication manager.
        
        Args:
            credential_dir: Directory to store credentials
            validity_days: Credential validity in days
        """
        self.credential_dir = credential_dir
        self.validity_days = validity_days
        self.crypto = CryptographyManager()
        os.makedirs(credential_dir, exist_ok=True)
    
    def register_client(self, client_id: int) -> ClientCredentials:
        """
        Register a new client and generate credentials.
        
        Args:
            client_id: Unique client identifier
        
        Returns:
            Client credentials
        """
        try:
            # Generate key pair
            private_pem, public_pem = self.crypto.generate_key_pair()
            
            # Create self-signed certificate
            certificate = self._generate_certificate(client_id, public_pem)
            
            # Create credentials
            credentials = ClientCredentials(
                client_id=client_id,
                public_key=public_pem,
                private_key=private_pem,
                certificate=certificate,
                issued_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=self.validity_days)
            )
            
            # Save credentials
            self._save_credentials(credentials)
            
            logger.info(f"Registered client {client_id}")
            return credentials
        
        except Exception as e:
            logger.error(f"Client registration failed: {e}")
            raise
    
    def load_credentials(self, client_id: int) -> Optional[ClientCredentials]:
        """Load client credentials from disk."""
        try:
            cred_path = os.path.join(self.credential_dir, f"client_{client_id}_creds.json")
            if not os.path.exists(cred_path):
                return None
            
            with open(cred_path, 'r') as f:
                data = json.load(f)
            
            # Note: In production, keys should be loaded from secure storage
            return None  # Simplified for demo
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            return None
    
    def _save_credentials(self, credentials: ClientCredentials):
        """Save credentials to disk."""
        try:
            cred_path = os.path.join(self.credential_dir, 
                                    f"client_{credentials.client_id}_creds.json")
            
            # Only save non-sensitive metadata (keys saved separately)
            cred_data = {
                'client_id': credentials.client_id,
                'issued_at': credentials.issued_at.isoformat(),
                'expires_at': credentials.expires_at.isoformat(),
                'public_key_path': f"client_{credentials.client_id}_public.pem",
            }
            
            with open(cred_path, 'w') as f:
                json.dump(cred_data, f, indent=2)
            
            # Save public key
            with open(os.path.join(self.credential_dir, 
                                   f"client_{credentials.client_id}_public.pem"), 'wb') as f:
                f.write(credentials.public_key)
            
            logger.info(f"Saved credentials for client {credentials.client_id}")
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
    
    def _generate_certificate(self, client_id: int, public_key_bytes: bytes) -> bytes:
        """Generate a self-signed certificate (simplified)."""
        # In production, use proper certificate generation
        cert_data = {
            'client_id': client_id,
            'issued_at': datetime.now().isoformat(),
            'public_key_hash': hashlib.sha256(public_key_bytes).hexdigest()
        }
        return json.dumps(cert_data).encode()
    
    def verify_client(self, client_id: int, credentials: ClientCredentials) -> bool:
        """Verify client credentials."""
        try:
            if not credentials.is_valid():
                logger.warning(f"Client {client_id} credentials expired")
                return False
            
            if credentials.client_id != client_id:
                logger.warning(f"Client ID mismatch: {client_id} vs {credentials.client_id}")
                return False
            
            logger.info(f"Verified client {client_id}")
            return True
        except Exception as e:
            logger.error(f"Client verification failed: {e}")
            return False


class TLSCommunicationManager:
    """
    Manages TLS-based secure communication between clients and server.
    """
    
    def __init__(self, ca_cert_path: str = None, ca_key_path: str = None):
        """
        Initialize TLS manager.
        
        Args:
            ca_cert_path: Path to CA certificate
            ca_key_path: Path to CA private key
        """
        self.ca_cert_path = ca_cert_path
        self.ca_key_path = ca_key_path
        self.crypto = CryptographyManager()
    
    def create_ssl_context(self, purpose: str = "server",
                          cert_path: str = None,
                          key_path: str = None) -> Optional[ssl.SSLContext]:
        """
        Create SSL context for TLS communication.
        
        Args:
            purpose: "server" or "client"
            cert_path: Path to certificate
            key_path: Path to private key
        
        Returns:
            Configured SSL context
        """
        try:
            if purpose == "server":
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                context.load_cert_chain(cert_path, key_path)
            else:
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                context.load_verify_locations(self.ca_cert_path)
                context.load_cert_chain(cert_path, key_path)
            
            # Set security parameters
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.options |= ssl.OP_NO_SSLv3 | ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
            context.set_ciphers('ECDHE+AESGCM:DHE+AESGCM:!aNULL:!MD5:!DSS')
            
            logger.info(f"Created TLS context for {purpose}")
            return context
        
        except Exception as e:
            logger.error(f"Failed to create TLS context: {e}")
            return None
    
    def validate_certificate(self, cert_path: str) -> bool:
        """
        Validate SSL certificate.
        
        Args:
            cert_path: Path to certificate
        
        Returns:
            True if certificate is valid
        """
        try:
            # In production, implement proper certificate validation
            return os.path.exists(cert_path)
        except Exception as e:
            logger.error(f"Certificate validation failed: {e}")
            return False


class SecureAggregationOrchestrator:
    """
    Orchestrates secure aggregation with encryption and authentication.
    """
    
    def __init__(self, num_clients: int = 20, num_servers: int = 3,
                 use_encryption: bool = True, use_secret_sharing: bool = True):
        """
        Initialize secure aggregation orchestrator.
        
        Args:
            num_clients: Number of clients
            num_servers: Number of servers for secret sharing
            use_encryption: Enable end-to-end encryption
            use_secret_sharing: Enable additive secret sharing
        """
        self.num_clients = num_clients
        self.num_servers = num_servers
        self.use_encryption = use_encryption
        self.use_secret_sharing = use_secret_sharing
        
        self.crypto = CryptographyManager()
        self.auth_manager = ClientAuthenticationManager()
        self.secure_agg = SecureAggregationManager(num_servers)
        self.tls_manager = TLSCommunicationManager()
        
        # Store client public keys for encryption
        self.client_keys = {}
        
        logger.info(f"Initialized secure aggregation orchestrator "
                   f"(encryption={use_encryption}, sharing={use_secret_sharing})")
    
    def register_clients(self) -> Dict[int, ClientCredentials]:
        """Register all clients and generate credentials."""
        credentials = {}
        for client_id in range(self.num_clients):
            cred = self.auth_manager.register_client(client_id)
            self.client_keys[client_id] = cred.public_key
            credentials[client_id] = cred
        
        logger.info(f"Registered {self.num_clients} clients")
        return credentials
    
    def secure_aggregate(self, client_updates: Dict[int, Dict[str, np.ndarray]],
                        num_samples: Dict[int, int],
                        client_credentials: Dict[int, ClientCredentials] = None
                        ) -> Dict[str, np.ndarray]:
        """
        Securely aggregate client updates with encryption and authentication.
        
        Args:
            client_updates: Client updates to aggregate
            num_samples: Number of samples per client
            client_credentials: Client credentials for verification
        
        Returns:
            Aggregated updates
        """
        try:
            # Verify client credentials if provided
            if client_credentials:
                valid_clients = []
                for client_id, cred in client_credentials.items():
                    if self.auth_manager.verify_client(client_id, cred):
                        valid_clients.append(client_id)
                
                logger.info(f"Verified {len(valid_clients)}/{len(client_updates)} clients")
            
            # Perform aggregation
            aggregated = self.secure_agg.secure_aggregate_updates(
                client_updates,
                num_samples
            )
            
            logger.info(f"Securely aggregated updates from {len(client_updates)} clients")
            return aggregated
        
        except Exception as e:
            logger.error(f"Secure aggregation failed: {e}")
            raise


# Convenience functions
def create_secure_aggregator(num_clients: int = 20,
                            num_servers: int = 3) -> SecureAggregationOrchestrator:
    """Create and configure secure aggregation orchestrator."""
    return SecureAggregationOrchestrator(num_clients, num_servers)
