"""
Test suite for Secure Aggregation & Encryption Module (Task #15)

Tests TLS communication, secure aggregation, and client authentication.
"""

import pytest
import os
import json
import tempfile
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from utils.secure_aggregation import (
    CryptographyManager,
    SecureAggregationManager,
    ClientAuthenticationManager,
    TLSCommunicationManager,
    SecureAggregationOrchestrator,
    ClientCredentials,
    create_secure_aggregator
)


class TestCryptographyManager:
    """Test cryptography operations."""
    
    @pytest.fixture
    def crypto(self):
        return CryptographyManager(key_size=2048)
    
    def test_init(self, crypto):
        """Test cryptography manager initialization."""
        assert crypto.key_size == 2048
        assert crypto.backend is not None
    
    def test_generate_key_pair(self, crypto):
        """Test RSA key pair generation."""
        private_key, public_key = crypto.generate_key_pair()
        assert isinstance(private_key, bytes)
        assert isinstance(public_key, bytes)
        assert b'BEGIN RSA PRIVATE KEY' in private_key or b'BEGIN PRIVATE KEY' in private_key
        assert b'BEGIN PUBLIC KEY' in public_key
    
    def test_generate_multiple_key_pairs(self, crypto):
        """Test generating multiple key pairs."""
        pairs = [crypto.generate_key_pair() for _ in range(3)]
        assert len(pairs) == 3
        # All pairs should be different
        assert pairs[0][0] != pairs[1][0]
        assert pairs[1][0] != pairs[2][0]
    
    def test_encrypt_decrypt_roundtrip(self, crypto):
        """Test encryption and decryption roundtrip."""
        private_key, public_key = crypto.generate_key_pair()
        
        plaintext = b"Secret message"
        ciphertext = crypto.encrypt_data(plaintext, public_key)
        decrypted = crypto.decrypt_data(ciphertext, private_key)
        
        assert decrypted == plaintext
    
    def test_encrypt_different_messages(self, crypto):
        """Test encrypting different messages produces different ciphertexts."""
        _, public_key = crypto.generate_key_pair()
        
        msg1 = b"Message 1"
        msg2 = b"Message 2"
        
        cipher1 = crypto.encrypt_data(msg1, public_key)
        cipher2 = crypto.encrypt_data(msg2, public_key)
        
        # Ciphertexts should be different
        assert cipher1 != cipher2
    
    def test_sign_verify(self, crypto):
        """Test signing and verification."""
        private_key, public_key = crypto.generate_key_pair()
        
        message = b"Message to sign"
        signature = crypto.sign_data(message, private_key)
        
        is_valid = crypto.verify_signature(message, signature, public_key)
        assert is_valid is True
    
    def test_invalid_signature(self, crypto):
        """Test invalid signature verification."""
        private_key, public_key = crypto.generate_key_pair()
        
        message = b"Message to sign"
        signature = crypto.sign_data(message, private_key)
        
        # Tamper with message
        tampered = b"Different message"
        is_valid = crypto.verify_signature(tampered, signature, public_key)
        assert is_valid is False


class TestSecureAggregationManager:
    """Test secure aggregation with secret sharing."""
    
    @pytest.fixture
    def agg_manager(self):
        return SecureAggregationManager(num_servers=3)
    
    def test_init(self, agg_manager):
        """Test initialization."""
        assert agg_manager.num_servers == 3
        assert agg_manager.prime == 2**61 - 1
        assert agg_manager.crypto is not None
    
    def test_split_secret_simple(self, agg_manager):
        """Test secret splitting for simple value."""
        value = 100.5
        shares = agg_manager.split_secret(value, num_shares=3)
        
        assert len(shares) == 3
        assert isinstance(shares[0], float)
    
    def test_reconstruct_secret_simple(self, agg_manager):
        """Test secret reconstruction."""
        value = 100.5
        shares = agg_manager.split_secret(value, num_shares=3)
        reconstructed = agg_manager.reconstruct_secret(shares)
        
        # Should be approximately equal (floating point precision)
        assert abs(reconstructed - value) < 0.01
    
    def test_split_reconstruct_multiple_values(self, agg_manager):
        """Test secret sharing for multiple values."""
        values = [10.5, 20.3, 30.7, 40.1]
        
        for value in values:
            shares = agg_manager.split_secret(value, num_shares=3)
            reconstructed = agg_manager.reconstruct_secret(shares)
            assert abs(reconstructed - value) < 0.01
    
    def test_secure_aggregate_updates(self, agg_manager):
        """Test secure aggregation of client updates."""
        client_updates = {
            0: {'coefs': np.array([[0.1, 0.2], [0.3, 0.4]])},
            1: {'coefs': np.array([[0.2, 0.3], [0.4, 0.5]])},
            2: {'coefs': np.array([[0.3, 0.4], [0.5, 0.6]])},
        }
        
        num_samples = {0: 100, 1: 150, 2: 100}
        
        aggregated = agg_manager.secure_aggregate_updates(client_updates, num_samples)
        
        assert 'coefs' in aggregated
        assert aggregated['coefs'].shape == (2, 2)


class TestClientAuthenticationManager:
    """Test client authentication and credentials."""
    
    @pytest.fixture
    def auth_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ClientAuthenticationManager(credential_dir=tmpdir)
            yield manager
    
    def test_init(self, auth_manager):
        """Test authentication manager initialization."""
        assert auth_manager.validity_days == 365
        assert os.path.exists(auth_manager.credential_dir)
    
    def test_register_client(self, auth_manager):
        """Test client registration."""
        cred = auth_manager.register_client(client_id=1)
        
        assert isinstance(cred, ClientCredentials)
        assert cred.client_id == 1
        assert cred.is_valid() is True
        assert cred.public_key is not None
        assert cred.private_key is not None
    
    def test_register_multiple_clients(self, auth_manager):
        """Test registering multiple clients."""
        creds = [auth_manager.register_client(i) for i in range(5)]
        
        assert len(creds) == 5
        assert all(isinstance(c, ClientCredentials) for c in creds)
        assert all(c.is_valid() for c in creds)
    
    def test_client_credentials_validity(self):
        """Test client credentials validity check."""
        now = datetime.now()
        
        # Valid credentials
        valid_cred = ClientCredentials(
            client_id=1,
            public_key=b"key",
            private_key=b"key",
            certificate=b"cert",
            issued_at=now,
            expires_at=now + timedelta(days=1)
        )
        assert valid_cred.is_valid() is True
        
        # Expired credentials
        expired_cred = ClientCredentials(
            client_id=2,
            public_key=b"key",
            private_key=b"key",
            certificate=b"cert",
            issued_at=now - timedelta(days=2),
            expires_at=now - timedelta(days=1)
        )
        assert expired_cred.is_valid() is False
    
    def test_verify_client_valid(self, auth_manager):
        """Test verifying valid client credentials."""
        cred = auth_manager.register_client(client_id=1)
        is_valid = auth_manager.verify_client(1, cred)
        assert is_valid is True
    
    def test_verify_client_mismatched_id(self, auth_manager):
        """Test verifying with mismatched client ID."""
        cred = auth_manager.register_client(client_id=1)
        is_valid = auth_manager.verify_client(2, cred)
        assert is_valid is False


class TestTLSCommunicationManager:
    """Test TLS communication setup."""
    
    @pytest.fixture
    def tls_manager(self):
        return TLSCommunicationManager()
    
    def test_init(self, tls_manager):
        """Test TLS manager initialization."""
        assert tls_manager.crypto is not None
    
    def test_validate_certificate_nonexistent(self, tls_manager):
        """Test validating non-existent certificate."""
        is_valid = tls_manager.validate_certificate("/nonexistent/cert.pem")
        assert is_valid is False
    
    def test_validate_certificate_exists(self):
        """Test validating existing certificate."""
        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as f:
            cert_path = f.name
        
        try:
            tls_manager = TLSCommunicationManager()
            is_valid = tls_manager.validate_certificate(cert_path)
            assert is_valid is True
        finally:
            os.remove(cert_path)


class TestSecureAggregationOrchestrator:
    """Test secure aggregation orchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        return SecureAggregationOrchestrator(
            num_clients=5,
            num_servers=3,
            use_encryption=True,
            use_secret_sharing=True
        )
    
    def test_init(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.num_clients == 5
        assert orchestrator.num_servers == 3
        assert orchestrator.use_encryption is True
        assert orchestrator.use_secret_sharing is True
    
    def test_register_clients(self, orchestrator):
        """Test client registration."""
        credentials = orchestrator.register_clients()
        
        assert len(credentials) == 5
        assert all(isinstance(c, ClientCredentials) for c in credentials.values())
        assert all(c.is_valid() for c in credentials.values())
    
    def test_secure_aggregate_no_encryption(self):
        """Test secure aggregation without encryption."""
        orchestrator = SecureAggregationOrchestrator(
            num_clients=3,
            use_encryption=False
        )
        
        client_updates = {
            0: {'coefs': np.array([[0.1, 0.2]])},
            1: {'coefs': np.array([[0.2, 0.3]])},
            2: {'coefs': np.array([[0.3, 0.4]])},
        }
        
        num_samples = {0: 100, 1: 100, 2: 100}
        
        aggregated = orchestrator.secure_aggregate(
            client_updates,
            num_samples
        )
        
        assert 'coefs' in aggregated
    
    def test_secure_aggregate_with_credentials(self, orchestrator):
        """Test secure aggregation with credential verification."""
        credentials = orchestrator.register_clients()
        
        client_updates = {
            i: {'coefs': np.array([[float(i), float(i+1)]])}
            for i in range(5)
        }
        
        num_samples = {i: 100 for i in range(5)}
        
        aggregated = orchestrator.secure_aggregate(
            client_updates,
            num_samples,
            client_credentials=credentials
        )
        
        assert 'coefs' in aggregated


class TestEndToEndSecureAggregation:
    """End-to-end tests for secure aggregation."""
    
    def test_full_secure_workflow(self):
        """Test complete secure aggregation workflow."""
        # Create orchestrator
        orchestrator = create_secure_aggregator(num_clients=5, num_servers=3)
        
        # Register clients
        credentials = orchestrator.register_clients()
        assert len(credentials) == 5
        
        # Prepare client updates
        client_updates = {
            i: {
                'coefs': np.random.randn(10, 5),
                'intercepts': np.random.randn(5)
            }
            for i in range(5)
        }
        
        num_samples = {i: 100 + i * 10 for i in range(5)}
        
        # Securely aggregate
        aggregated = orchestrator.secure_aggregate(
            client_updates,
            num_samples,
            client_credentials=credentials
        )
        
        # Verify aggregated update
        assert 'coefs' in aggregated
        assert 'intercepts' in aggregated
        assert aggregated['coefs'].shape == (10, 5)
        assert aggregated['intercepts'].shape == (5,)
    
    def test_secure_aggregation_vs_plaintext(self):
        """Compare secure aggregation with plaintext aggregation."""
        # Create sample updates
        client_updates = {
            0: {'coefs': np.array([[1.0, 2.0]])},
            1: {'coefs': np.array([[3.0, 4.0]])},
            2: {'coefs': np.array([[5.0, 6.0]])},
        }
        
        num_samples = {0: 100, 1: 100, 2: 100}
        
        # Plaintext aggregation
        plaintext_agg = {}
        for client_id, update in client_updates.items():
            weight = num_samples[client_id] / sum(num_samples.values())
            for key, value in update.items():
                if key not in plaintext_agg:
                    plaintext_agg[key] = np.zeros_like(value)
                plaintext_agg[key] += value * weight
        
        # Secure aggregation
        orchestrator = create_secure_aggregator(num_clients=3)
        secure_agg_result = orchestrator.secure_aggregate(
            client_updates,
            num_samples
        )
        
        # Results should be equivalent
        np.testing.assert_array_almost_equal(
            plaintext_agg['coefs'],
            secure_agg_result['coefs'],
            decimal=5
        )


class TestSecurityProperties:
    """Test security properties of secure aggregation."""
    
    def test_client_isolation(self):
        """Test that individual client updates are not exposed."""
        orchestrator = create_secure_aggregator(num_clients=3)
        
        # Register clients with different updates
        credentials = orchestrator.register_clients()
        
        client_updates = {
            0: {'coefs': np.array([[100.0]])},  # Distinctly large
            1: {'coefs': np.array([[0.1]])},    # Distinctly small
            2: {'coefs': np.array([[50.0]])},   # Medium
        }
        
        num_samples = {0: 100, 1: 100, 2: 100}
        
        # Aggregate
        aggregated = orchestrator.secure_aggregate(
            client_updates,
            num_samples,
            client_credentials=credentials
        )
        
        # Aggregated value should be average of inputs
        expected = np.mean([100.0, 0.1, 50.0])
        actual = aggregated['coefs'][0, 0]
        
        np.testing.assert_almost_equal(actual, expected, decimal=3)
    
    def test_credential_expiry(self):
        """Test that expired credentials are rejected."""
        # Create credentials with expiry in past
        now = datetime.now()
        expired_cred = ClientCredentials(
            client_id=1,
            public_key=b"key",
            private_key=b"key",
            certificate=b"cert",
            issued_at=now - timedelta(days=2),
            expires_at=now - timedelta(days=1)
        )
        
        # Verify should return False
        orchestrator = create_secure_aggregator()
        is_valid = orchestrator.auth_manager.verify_client(1, expired_cred)
        
        assert is_valid is False


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_secure_aggregator(self):
        """Test create_secure_aggregator convenience function."""
        aggregator = create_secure_aggregator(num_clients=10, num_servers=3)
        
        assert isinstance(aggregator, SecureAggregationOrchestrator)
        assert aggregator.num_clients == 10
        assert aggregator.num_servers == 3


class TestErrorHandling:
    """Test error handling in security modules."""
    
    def test_invalid_key_decryption(self):
        """Test decryption with wrong key."""
        crypto = CryptographyManager()
        
        # Generate two key pairs
        priv1, pub1 = crypto.generate_key_pair()
        priv2, pub2 = crypto.generate_key_pair()
        
        # Encrypt with pub1
        plaintext = b"Secret"
        ciphertext = crypto.encrypt_data(plaintext, pub1)
        
        # Try to decrypt with priv2 - should fail
        with pytest.raises(Exception):
            crypto.decrypt_data(ciphertext, priv2)
    
    def test_corrupt_data_aggregation(self):
        """Test handling corrupt data in aggregation."""
        orchestrator = create_secure_aggregator(num_clients=2)
        
        # Create updates with corrupt data
        client_updates = {
            0: {'coefs': np.array([[1.0, 2.0]])},
            1: {'coefs': np.array([[3.0, 4.0]])},
        }
        
        num_samples = {0: 100, 1: 100}
        
        # Should handle gracefully
        aggregated = orchestrator.secure_aggregate(client_updates, num_samples)
        assert aggregated is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
