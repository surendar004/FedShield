"""
Comprehensive test suite for FedShield federated learning system.
Tests preprocessing, models, and federated algorithms.
"""
import pytest
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from client.preprocessor import FeaturePreprocessor
from client.model import ThreatDetectionModel, ServerEnsemble
from server.federated_learning import FLConfig, FederatedServer, FederatedClient, FedAvgOrchestrator


class TestPreprocessor:
    """Test feature preprocessing pipeline."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor can be initialized."""
        preprocessor = FeaturePreprocessor()
        assert preprocessor.is_fitted == False
        assert len(preprocessor.feature_names) == 27
        assert len(preprocessor.label_mapping) == 6
    
    def test_fit_transform(self):
        """Test fit_transform normalization."""
        preprocessor = FeaturePreprocessor()
        X = np.random.randn(100, 27).astype(np.float32)
        
        X_norm = preprocessor.fit_transform(X)
        
        assert X_norm.shape == X.shape
        assert preprocessor.is_fitted == True
        # Check normalization
        assert np.abs(X_norm.mean()) < 0.1  # Close to 0
        assert np.abs(X_norm.std() - 1.0) < 0.1  # Close to 1
    
    def test_inverse_transform(self):
        """Test inverse normalization."""
        preprocessor = FeaturePreprocessor()
        X_original = np.random.randn(50, 27).astype(np.float32)
        
        X_norm = preprocessor.fit_transform(X_original)
        X_reconstructed = preprocessor.inverse_transform(X_norm)
        
        np.testing.assert_array_almost_equal(X_original, X_reconstructed, decimal=5)
    
    def test_label_mapping(self):
        """Test label encoding/decoding."""
        preprocessor = FeaturePreprocessor()
        assert preprocessor.label_mapping['NORMAL'] == 0
        assert preprocessor.label_mapping['MALWARE'] == 1
        assert preprocessor.label_mapping['PHISHING'] == 2
        assert len(preprocessor.label_mapping) == 6


class TestThreatDetectionModel:
    """Test threat detection model."""
    
    def test_model_build(self):
        """Test model initialization."""
        model = ThreatDetectionModel()
        assert model.is_fitted == False
        
        model.build()
        assert model.model is not None
    
    def test_model_training(self):
        """Test local training."""
        model = ThreatDetectionModel(max_iter=10)
        X = np.random.randn(200, 27).astype(np.float32)
        y = np.random.randint(0, 6, 200)
        
        metrics = model.train(X, y)
        
        assert model.is_fitted == True
        assert 'accuracy' in metrics
        assert metrics['accuracy'] >= 0.0 and metrics['accuracy'] <= 1.0
    
    def test_model_prediction(self):
        """Test predictions."""
        model = ThreatDetectionModel(max_iter=10)
        X_train = np.random.randn(100, 27).astype(np.float32)
        y_train = np.random.randint(0, 6, 100)
        model.train(X_train, y_train)
        
        X_test = np.random.randn(20, 27).astype(np.float32)
        preds = model.predict(X_test)
        
        assert preds.shape == (20,)
        assert all(p in range(6) for p in preds)
    
    def test_predict_proba(self):
        """Test probability predictions."""
        model = ThreatDetectionModel(max_iter=10)
        X_train = np.random.randn(100, 27).astype(np.float32)
        y_train = np.random.randint(0, 6, 100)
        model.train(X_train, y_train)
        
        X_test = np.random.randn(10, 27).astype(np.float32)
        proba = model.predict_proba(X_test)
        
        assert proba.shape == (10, 6)
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1
    
    def test_get_set_weights(self):
        """Test weight extraction and setting."""
        model1 = ThreatDetectionModel(max_iter=5)
        X = np.random.randn(50, 27).astype(np.float32)
        y = np.random.randint(0, 6, 50)
        model1.train(X, y)
        
        weights = model1.get_weights()
        assert 'coefs' in weights
        assert 'intercepts' in weights
        assert 'scaler_mean' in weights
        assert 'scaler_scale' in weights
        
        # Test setting weights
        model2 = ThreatDetectionModel()
        model2.set_weights(weights)
        assert model2.is_fitted == True


class TestServerEnsemble:
    """Test server-side ensemble aggregation."""
    
    def test_ensemble_aggregation(self):
        """Test weighted aggregation of client models."""
        # Create mock client updates
        client_updates = {}
        for cid in range(3):
            client_updates[cid] = {
                'coefs': [np.random.randn(27, 128), np.random.randn(128, 64), np.random.randn(64, 32)],
                'intercepts': [np.random.randn(128), np.random.randn(64), np.random.randn(32)],
                'scaler_mean': np.random.randn(27),
                'scaler_scale': np.random.randn(27)
            }
        
        ensemble = ServerEnsemble(num_clients=3)
        weights = {0: 100, 1: 150, 2: 50}  # Sample counts
        
        aggregated = ensemble.aggregate(client_updates, weights)
        
        assert 'coefs' in aggregated
        assert 'intercepts' in aggregated
        assert len(aggregated['coefs']) == 3
        assert len(aggregated['intercepts']) == 3


class TestFLConfig:
    """Test FL configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = FLConfig()
        assert config.num_rounds == 10
        assert config.clients_per_round == 3
        assert config.local_epochs == 5
    
    def test_config_to_dict(self):
        """Test config serialization."""
        config = FLConfig(num_rounds=5, local_epochs=3)
        config_dict = config.to_dict()
        
        assert config_dict['num_rounds'] == 5
        assert config_dict['local_epochs'] == 3
        assert isinstance(config_dict, dict)


class TestFederatedServer:
    """Test server-side FL operations."""
    
    def test_client_selection(self):
        """Test client sampling."""
        config = FLConfig(clients_per_round=2)
        server = FederatedServer(config, num_clients=5)
        
        selected = server.client_selection()
        assert len(selected) <= config.clients_per_round
        assert all(cid < 5 for cid in selected)
    
    def test_aggregate_updates(self):
        """Test weight aggregation."""
        config = FLConfig()
        server = FederatedServer(config, num_clients=3)
        
        # Create mock updates
        client_updates = {}
        for cid in range(2):
            client_updates[cid] = {
                'coefs': [np.ones((27, 128)), np.ones((128, 64)), np.ones((64, 32))],
                'intercepts': [np.ones(128), np.ones(64), np.ones(32)],
                'scaler_mean': np.ones(27),
                'scaler_scale': np.ones(27)
            }
        
        num_samples = {0: 100, 1: 150}
        aggregated = server.aggregate_updates(client_updates, num_samples)
        
        assert 'coefs' in aggregated
        assert np.allclose(aggregated['coefs'][0], 1.0)  # Uniform weights should give 1.0


class TestFederatedClient:
    """Test client-side FL operations."""
    
    def test_client_initialization(self):
        """Test client setup."""
        client = FederatedClient(client_id=0, local_data_size=100)
        assert client.client_id == 0
        assert client.is_fitted == False if hasattr(client, 'is_fitted') else True
    
    def test_receive_global_weights(self):
        """Test receiving model from server."""
        client = FederatedClient(0, 100)
        model = ThreatDetectionModel(max_iter=5)
        client.set_model(model)
        
        X = np.random.randn(50, 27).astype(np.float32)
        y = np.random.randint(0, 6, 50)
        model.train(X, y)
        
        weights = model.get_weights()
        client.receive_global_weights(weights)
        
        assert client.global_model_weights is not None


class TestFedAvgOrchestrator:
    """Test full FedAvg training orchestration."""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator setup."""
        config = FLConfig(num_rounds=2, clients_per_round=2)
        orchestrator = FedAvgOrchestrator(config, num_clients=3)
        
        assert orchestrator.server.num_clients == 3
        assert len(orchestrator.clients) == 3
    
    def test_simulate_round(self):
        """Test one FL round simulation."""
        config = FLConfig(num_rounds=1, clients_per_round=2, local_epochs=1)
        orchestrator = FedAvgOrchestrator(config, num_clients=3)
        
        # Setup models for each client
        for cid in orchestrator.clients.keys():
            orchestrator.clients[cid].set_model(ThreatDetectionModel(max_iter=5))
        
        # Create mock data
        client_data = {}
        num_samples = {}
        for cid in range(2):  # 2 clients
            X = np.random.randn(50, 27).astype(np.float32)
            y = np.random.randint(0, 6, 50)
            client_data[cid] = (X, y)
            num_samples[cid] = 50
        
        # Run round
        summary = orchestrator.simulate_round(client_data, num_samples)
        
        assert 'round' in summary
        assert summary['num_updates'] > 0


class TestNonIIDSimulation:
    """Test non-IID data distribution simulation."""
    
    def test_skewed_distribution(self):
        """Test creating skewed label distribution."""
        n_samples = 100
        n_classes = 6
        imbalance_factor = 0.8
        
        # Create skewed distribution
        probs = np.ones(n_classes) / n_classes
        probs[0] = imbalance_factor  # Dominant class
        probs = probs / probs.sum()
        
        labels = np.random.choice(n_classes, n_samples, p=probs)
        
        # Check imbalance
        assert labels.tolist().count(0) > labels.tolist().count(1)
        assert np.sum(labels == 0) / n_samples > 0.1


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_fl_simulation(self):
        """Test complete FL training loop."""
        config = FLConfig(num_rounds=2, clients_per_round=2, local_epochs=1)
        orchestrator = FedAvgOrchestrator(config, num_clients=3)
        
        # Setup
        for cid in orchestrator.clients.keys():
            model = ThreatDetectionModel(max_iter=5)
            orchestrator.clients[cid].set_model(model)
        
        # Create data
        np.random.seed(42)
        client_data = {}
        num_samples = {}
        for cid in range(3):
            X = np.random.randn(30, 27).astype(np.float32)
            y = np.random.randint(0, 6, 30)
            client_data[cid] = (X, y)
            num_samples[cid] = 30
        
        # Run FL rounds
        for _ in range(config.num_rounds):
            summary = orchestrator.simulate_round(client_data, num_samples)
            assert summary is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
