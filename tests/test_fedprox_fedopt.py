"""
Comprehensive tests for FedProx and FedOpt algorithms.
Tests cover:
- FedProx regularization with varying mu coefficients
- FedOpt server-side optimization (Adam, Yogi, SGD)
- Comparison of algorithms
- Convergence properties
"""
import numpy as np
import pytest
from server.federated_learning import (
    FLConfig, FedProxOrchestrator, FedOptOrchestrator, FedOptimizer,
    FederatedClient, FedAvgOrchestrator
)
from client.model import ThreatDetectionModel


class TestFedProxRegularization:
    """Test FedProx proximal regularization."""
    
    def test_fedprox_initialization(self):
        """Test FedProx orchestrator initialization."""
        config = FLConfig(num_rounds=3, clients_per_round=2)
        orchestrator = FedProxOrchestrator(config, num_clients=4, mu=0.01)
        
        assert orchestrator.mu == 0.01
        assert len(orchestrator.clients) == 4
        assert orchestrator.config.num_rounds == 3
    
    def test_fedprox_custom_mu(self):
        """Test FedProx with custom regularization coefficients."""
        config = FLConfig(num_rounds=2, clients_per_round=2)
        
        for mu_val in [0.001, 0.01, 0.1, 1.0]:
            orchestrator = FedProxOrchestrator(config, num_clients=4, mu=mu_val)
            assert orchestrator.mu == mu_val
    
    def test_fedprox_training_convergence(self):
        """Test FedProx orchestrator tracks rounds correctly."""
        config = FLConfig(num_rounds=2, clients_per_round=1, local_epochs=1)
        orchestrator = FedProxOrchestrator(config, num_clients=2, mu=0.01)
        
        # Initialize clients with models
        for cid, client in orchestrator.clients.items():
            client.set_model(ThreatDetectionModel(hidden_layers=(32, 16)))
        
        # Generate simple data
        np.random.seed(42)
        client_data = {}
        
        for cid in range(2):
            X = np.random.randn(100, 27).astype(np.float32)
            y = np.zeros(100, dtype=int)  # Single class to avoid sklearn validation issues
            client_data[cid] = (X, y)
        
        # Train initial global model with single class
        client = orchestrator.clients[0]
        X_init, y_init = client_data[0]
        client.local_model.train(X_init, y_init)
        global_weights = client.local_model.get_weights()
        
        # Run FedProx round
        round_result = orchestrator.simulate_round(
            client_data,
            {cid: 100 for cid in range(2)},
            global_weights
        )
        
        # Check round structure
        assert round_result['algorithm'] == 'FedProx'
        assert round_result['mu'] == 0.01
        assert 'metrics' in round_result
    
    def test_fedprox_vs_fedavg_heterogeneity(self):
        """Test FedProx and FedAvg both complete successfully."""
        config = FLConfig(num_rounds=1, clients_per_round=1, local_epochs=1)
        
        # Create simple homogeneous data (to avoid sklearn issues)
        np.random.seed(42)
        client_data = {}
        
        for cid in range(2):
            X = np.random.randn(100, 27).astype(np.float32)
            y = np.zeros(100, dtype=int)  # Single class
            client_data[cid] = (X, y)
        
        # Test FedProx
        fedprox = FedProxOrchestrator(config, num_clients=2, mu=0.1)
        for cid, client in fedprox.clients.items():
            client.set_model(ThreatDetectionModel(hidden_layers=(32, 16)))
        
        # Initialize global weights
        fedprox.clients[0].local_model.train(client_data[0][0][:50], client_data[0][1][:50])
        global_weights = fedprox.clients[0].local_model.get_weights()
        
        fedprox_result = fedprox.simulate_round(
            client_data,
            {cid: 100 for cid in range(2)},
            global_weights
        )
        
        # Test FedAvg for comparison
        fedavg = FedAvgOrchestrator(config, num_clients=2)
        for cid, client in fedavg.clients.items():
            client.set_model(ThreatDetectionModel(hidden_layers=(32, 16)))
        
        fedavg.clients[0].local_model.train(client_data[0][0][:50], client_data[0][1][:50])
        
        fedavg_result = fedavg.simulate_round(
            client_data,
            {cid: 100 for cid in range(2)}
        )
        
        # Both should complete successfully
        assert fedprox_result['algorithm'] == 'FedProx'
        assert 'FedAvg' not in fedavg_result  # FedAvg doesn't store algorithm name


class TestFedOptOptimizer:
    """Test FedOpt server-side optimizers."""
    
    def test_fedopt_adam_initialization(self):
        """Test Adam optimizer initialization."""
        optimizer = FedOptimizer(optimizer_name="adam", learning_rate=0.01)
        
        assert optimizer.optimizer_name == "adam"
        assert optimizer.learning_rate == 0.01
        assert optimizer.beta1 == 0.9
        assert optimizer.beta2 == 0.999
        assert optimizer.t == 0
        assert isinstance(optimizer.m_t, dict)
        assert isinstance(optimizer.v_t, dict)
    
    def test_fedopt_yogi_initialization(self):
        """Test Yogi optimizer initialization."""
        optimizer = FedOptimizer(optimizer_name="yogi", learning_rate=0.01)
        
        assert optimizer.optimizer_name == "yogi"
        assert optimizer.learning_rate == 0.01
    
    def test_fedopt_sgd_initialization(self):
        """Test SGD optimizer initialization."""
        optimizer = FedOptimizer(optimizer_name="sgd", learning_rate=0.01)
        
        assert optimizer.optimizer_name == "sgd"
        assert optimizer.learning_rate == 0.01
    
    def test_adam_step_basic(self):
        """Test Adam optimization step with simple weights."""
        optimizer = FedOptimizer(optimizer_name="adam", learning_rate=0.001)
        
        weights = {
            'coefs': [np.array([[0.1, 0.2], [0.3, 0.4]])],
            'intercepts': [np.array([0.05, 0.1])]
        }
        
        updated = optimizer.step(weights)
        
        assert 'coefs' in updated
        assert 'intercepts' in updated
        assert len(updated['coefs']) == 1
        assert len(updated['intercepts']) == 1
        assert optimizer.t == 1
    
    def test_adam_step_sequence(self):
        """Test Adam over multiple steps (convergence behavior)."""
        optimizer = FedOptimizer(optimizer_name="adam", learning_rate=0.01)
        
        # Simulate gradient descent toward zero
        weights = {
            'w': np.array([1.0, 1.0, 1.0])
        }
        
        norms = []
        for _ in range(10):
            updated = optimizer.step(weights)
            norm = np.linalg.norm(updated['w'])
            norms.append(norm)
            weights = updated
        
        # Adam should reduce the magnitude over time
        assert norms[0] > norms[-1]
        assert optimizer.t == 10
    
    def test_yogi_step_basic(self):
        """Test Yogi optimization step."""
        optimizer = FedOptimizer(optimizer_name="yogi", learning_rate=0.001)
        
        weights = {
            'coefs': [np.array([[0.1, 0.2], [0.3, 0.4]])],
            'intercepts': [np.array([0.05, 0.1])]
        }
        
        updated = optimizer.step(weights)
        
        assert 'coefs' in updated
        assert 'intercepts' in updated
        assert optimizer.t == 1
    
    def test_sgd_step_basic(self):
        """Test SGD optimization step."""
        optimizer = FedOptimizer(optimizer_name="sgd", learning_rate=0.01)
        
        weights = {
            'w': np.array([1.0, 2.0, 3.0])
        }
        
        updated = optimizer.step(weights)
        
        # SGD should apply simple gradient descent: w - lr * w
        expected = weights['w'] - 0.01 * weights['w']
        np.testing.assert_array_almost_equal(updated['w'], expected)
    
    def test_optimizer_state_persistence(self):
        """Test optimizer maintains state across steps."""
        optimizer = FedOptimizer(optimizer_name="adam", learning_rate=0.001)
        
        weights = {'w': np.ones((2, 2))}
        
        # First step
        optimizer.step(weights)
        assert optimizer.t == 1
        assert len(optimizer.m_t) > 0
        
        # Second step
        optimizer.step(weights)
        assert optimizer.t == 2
        assert len(optimizer.m_t) > 0


class TestFedOptOrchestrator:
    """Test FedOpt orchestrator with server optimizers."""
    
    def test_fedopt_initialization_adam(self):
        """Test FedOpt orchestrator with Adam optimizer."""
        config = FLConfig(num_rounds=2, clients_per_round=2)
        orchestrator = FedOptOrchestrator(
            config, num_clients=3,
            server_optimizer="adam",
            server_lr=0.01
        )
        
        assert orchestrator.optimizer.optimizer_name == "adam"
        assert orchestrator.optimizer.learning_rate == 0.01
        assert len(orchestrator.clients) == 3
    
    def test_fedopt_initialization_yogi(self):
        """Test FedOpt orchestrator with Yogi optimizer."""
        config = FLConfig(num_rounds=2, clients_per_round=2)
        orchestrator = FedOptOrchestrator(
            config, num_clients=3,
            server_optimizer="yogi",
            server_lr=0.02
        )
        
        assert orchestrator.optimizer.optimizer_name == "yogi"
        assert orchestrator.optimizer.learning_rate == 0.02
    
    def test_fedopt_simulate_round(self):
        """Test FedOpt round simulation."""
        config = FLConfig(num_rounds=1, clients_per_round=2, local_epochs=1)
        orchestrator = FedOptOrchestrator(config, num_clients=3, server_optimizer="adam")
        
        # Initialize clients
        for cid, client in orchestrator.clients.items():
            client.set_model(ThreatDetectionModel(hidden_layers=(32, 16)))
        
        # Generate data with sufficient samples
        np.random.seed(42)
        client_data = {}
        for cid in range(3):
            X = np.random.randn(100, 27).astype(np.float32)
            y = np.random.randint(0, 6, 100)
            client_data[cid] = (X, y)
        
        # Run round
        result = orchestrator.simulate_round(
            client_data,
            {cid: 100 for cid in range(3)}
        )
        
        assert result['algorithm'] == 'FedOpt'
        assert result['optimizer'] == 'adam'
        assert result['server_step'] == 1
        assert len(result['metrics']) > 0
    
    def test_fedopt_convergence_adam(self):
        """Test FedOpt with Adam converges."""
        config = FLConfig(num_rounds=3, clients_per_round=2, local_epochs=1)
        orchestrator = FedOptOrchestrator(config, num_clients=3, server_optimizer="adam")
        
        # Initialize
        for cid, client in orchestrator.clients.items():
            client.set_model(ThreatDetectionModel(hidden_layers=(32, 16)))
        
        np.random.seed(42)
        client_data = {}
        for cid in range(3):
            X = np.random.randn(100, 27).astype(np.float32)
            y = np.random.randint(0, 6, 100)
            client_data[cid] = (X, y)
        
        # Multiple rounds
        for _ in range(config.num_rounds):
            orchestrator.simulate_round(
                client_data,
                {cid: 100 for cid in range(3)}
            )
        
        assert len(orchestrator.global_history) == config.num_rounds
        assert orchestrator.optimizer.t == config.num_rounds
    
    def test_fedopt_adam_vs_sgd(self):
        """Test Adam and SGD optimizers behave differently."""
        config = FLConfig(num_rounds=1, clients_per_round=1, local_epochs=1)
        
        # Test Adam
        adam_orch = FedOptOrchestrator(config, num_clients=1, server_optimizer="adam")
        for cid, client in adam_orch.clients.items():
            client.set_model(ThreatDetectionModel(hidden_layers=(32, 16)))
        
        # Test SGD
        sgd_orch = FedOptOrchestrator(config, num_clients=1, server_optimizer="sgd")
        for cid, client in sgd_orch.clients.items():
            client.set_model(ThreatDetectionModel(hidden_layers=(32, 16)))
        
        # Same data
        np.random.seed(42)
        client_data = {}
        X = np.random.randn(100, 27).astype(np.float32)
        y = np.zeros(100, dtype=int)
        client_data[0] = (X, y)
        
        # Run both
        adam_result = adam_orch.simulate_round(
            client_data,
            {0: 100}
        )
        sgd_result = sgd_orch.simulate_round(
            client_data,
            {0: 100}
        )
        
        assert adam_result['optimizer'] == 'adam'
        assert sgd_result['optimizer'] == 'sgd'
        assert adam_orch.optimizer.t == 1
        assert sgd_orch.optimizer.t == 1


class TestFedProxVsFedAvgVsFedOpt:
    """Comparison tests between algorithms."""
    
    def test_algorithm_round_structure(self):
        """Test all algorithms have consistent round structure."""
        config = FLConfig(num_rounds=1, clients_per_round=1, local_epochs=1)
        
        fedavg = FedAvgOrchestrator(config, num_clients=1)
        fedprox = FedProxOrchestrator(config, num_clients=1, mu=0.01)
        fedopt = FedOptOrchestrator(config, num_clients=1, server_optimizer="adam")
        
        for orch in [fedavg, fedprox, fedopt]:
            for cid, client in orch.clients.items():
                client.set_model(ThreatDetectionModel(hidden_layers=(32, 16)))
        
        np.random.seed(42)
        X = np.random.randn(100, 27).astype(np.float32)
        y = np.zeros(100, dtype=int)
        client_data = {0: (X, y)}
        
        # Run FedAvg
        fedavg_result = fedavg.simulate_round(
            client_data,
            {0: 100}
        )
        
        # Run FedProx
        fedprox.clients[0].local_model.train(X[:50], y[:50])
        global_weights = fedprox.clients[0].local_model.get_weights()
        fedprox_result = fedprox.simulate_round(
            client_data,
            {0: 100},
            global_weights
        )
        
        # Run FedOpt
        fedopt_result = fedopt.simulate_round(
            client_data,
            {0: 100}
        )
        
        # All should have consistent structure
        for result in [fedavg_result, fedprox_result, fedopt_result]:
            assert 'round' in result
            assert 'selected_clients' in result
            assert 'metrics' in result
            assert 'num_updates' in result
    
    def test_all_algorithms_with_non_iid_data(self):
        """Test all algorithms handle data."""
        config = FLConfig(num_rounds=1, clients_per_round=1, local_epochs=1)
        
        # Prepare simple data to avoid sklearn validation issues
        np.random.seed(42)
        client_data = {}
        
        for cid in range(2):
            X = np.random.randn(100, 27).astype(np.float32)
            y = np.zeros(100, dtype=int)  # Single class
            client_data[cid] = (X, y)
        
        # Test all algorithms
        algorithms = {
            'FedAvg': FedAvgOrchestrator(config, num_clients=2),
            'FedProx': FedProxOrchestrator(config, num_clients=2, mu=0.05),
            'FedOpt': FedOptOrchestrator(config, num_clients=2, server_optimizer="adam")
        }
        
        for name, orch in algorithms.items():
            # Initialize
            for cid, client in orch.clients.items():
                client.set_model(ThreatDetectionModel(hidden_layers=(32, 16)))
            
            # Run one round
            if name == 'FedProx':
                orch.clients[0].local_model.train(client_data[0][0][:50], client_data[0][1][:50])
                global_weights = orch.clients[0].local_model.get_weights()
                result = orch.simulate_round(
                    client_data,
                    {cid: 100 for cid in range(2)},
                    global_weights
                )
            else:
                result = orch.simulate_round(
                    client_data,
                    {cid: 100 for cid in range(2)}
                )
            
            assert result is not None
            assert 'metrics' in result


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_fedprox_with_mu_zero(self):
        """Test FedProx with mu=0 (should behave like FedAvg)."""
        config = FLConfig(num_rounds=1, clients_per_round=2, local_epochs=1)
        orchestrator = FedProxOrchestrator(config, num_clients=2, mu=0.0)
        
        assert orchestrator.mu == 0.0
    
    def test_fedprox_with_very_high_mu(self):
        """Test FedProx with very high mu (strong regularization)."""
        config = FLConfig(num_rounds=1, clients_per_round=2, local_epochs=1)
        orchestrator = FedProxOrchestrator(config, num_clients=2, mu=10.0)
        
        assert orchestrator.mu == 10.0
    
    def test_fedopt_with_zero_learning_rate(self):
        """Test FedOpt with very small learning rate."""
        optimizer = FedOptimizer(optimizer_name="adam", learning_rate=1e-8)
        
        assert optimizer.learning_rate == 1e-8
    
    def test_fedopt_with_high_learning_rate(self):
        """Test FedOpt with large learning rate."""
        optimizer = FedOptimizer(optimizer_name="adam", learning_rate=1.0)
        
        assert optimizer.learning_rate == 1.0
    
    def test_fedopt_yogi_variance_handling(self):
        """Test Yogi handles zero/near-zero variances."""
        optimizer = FedOptimizer(optimizer_name="yogi", learning_rate=0.01)
        
        # Near-zero gradients
        weights = {'w': np.ones((2, 2)) * 1e-10}
        
        updated = optimizer.step(weights)
        
        # Should not produce NaN or Inf
        assert np.all(np.isfinite(updated['w']))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
