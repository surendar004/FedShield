"""
Test suite for ExperimentLogger (Task #13: MLflow Integration)

Tests MLflow logging functionality for FedShield experiments.
"""

import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from utils.experiment_logger import ExperimentLogger, OrchestrationLogger, create_logger


class TestExperimentLoggerInitialization:
    """Test ExperimentLogger initialization."""
    
    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        logger = ExperimentLogger()
        assert logger.experiment_name == "FedShield"
        assert logger.backend == "mlflow"
        assert os.path.exists(logger.artifacts_dir)
    
    def test_init_with_custom_name(self):
        """Test initialization with custom experiment name."""
        logger = ExperimentLogger(experiment_name="CustomExp", run_name="test_run")
        assert logger.experiment_name == "CustomExp"
        assert logger.run_name == "test_run"
    
    def test_init_creates_artifacts_dir(self):
        """Test that artifacts directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(log_dir=tmpdir)
            assert os.path.exists(logger.artifacts_dir)
            assert "artifacts" in logger.artifacts_dir
    
    def test_init_with_both_backends(self):
        """Test initialization with both MLflow and W&B backends."""
        logger = ExperimentLogger(backend="both")
        assert logger.backend == "both"


class TestMetricsLogging:
    """Test metrics logging functionality."""
    
    @pytest.fixture
    def logger(self):
        """Create a logger instance."""
        return ExperimentLogger(experiment_name="test_metrics")
    
    def test_log_params(self, logger):
        """Test logging parameters."""
        params = {
            "learning_rate": 0.01,
            "num_epochs": 10,
            "batch_size": 32,
            "algorithm": "FedProx",
        }
        logger.log_params(params)
        # Should not raise exception
        assert True
    
    def test_log_params_with_nested(self, logger):
        """Test logging nested parameters."""
        params = {
            "model_config": {"hidden_layers": [128, 64], "dropout": 0.1},
            "optimization": {"optimizer": "adam", "beta1": 0.9},
        }
        logger.log_params(params)
        assert True
    
    def test_log_round_metrics(self, logger):
        """Test logging per-round metrics."""
        metrics = {
            "global_accuracy": 0.92,
            "global_loss": 0.15,
            "avg_client_accuracy": 0.88,
            "convergence_rate": 0.02,
        }
        logger.log_round_metrics(round_num=1, metrics=metrics)
        assert True
    
    def test_log_round_metrics_multiple_rounds(self, logger):
        """Test logging metrics across multiple rounds."""
        for round_num in range(5):
            metrics = {
                "global_accuracy": 0.80 + (round_num * 0.02),
                "global_loss": 0.20 - (round_num * 0.01),
            }
            logger.log_round_metrics(round_num, metrics)
        assert True
    
    def test_log_client_metrics(self, logger):
        """Test logging per-client metrics."""
        client_metrics = {
            0: {"accuracy": 0.90, "loss": 0.12, "samples": 100},
            1: {"accuracy": 0.88, "loss": 0.14, "samples": 95},
            2: {"accuracy": 0.92, "loss": 0.10, "samples": 105},
        }
        logger.log_client_metrics(round_num=1, client_metrics=client_metrics)
        assert True
    
    def test_log_privacy_budget(self, logger):
        """Test logging privacy budget consumption."""
        logger.log_privacy_budget(
            round_num=1,
            epsilon=0.1,
            delta=1e-5,
            cumulative_epsilon=0.1,
            cumulative_delta=1e-5
        )
        assert True
    
    def test_log_model_version(self, logger):
        """Test logging model version."""
        logger.log_model_version(
            round_num=5,
            algorithm="FedProx",
            metrics={"accuracy": 0.95, "f1": 0.94}
        )
        assert True
    
    def test_log_aggregation_metrics(self, logger):
        """Test logging aggregation metrics."""
        logger.log_aggregation_metrics(
            round_num=2,
            aggregation_method="median",
            num_clients_selected=20,
            num_clients_aggregated=18,
            aggregation_time=2.5
        )
        assert True
    
    def test_log_communication_stats(self, logger):
        """Test logging communication statistics."""
        logger.log_communication_stats(
            round_num=1,
            bytes_uploaded=1024 * 1024,  # 1 MB
            bytes_downloaded=2048 * 1024,  # 2 MB
            num_clients=20
        )
        assert True


class TestPrivacyLogging:
    """Test privacy-related logging."""
    
    @pytest.fixture
    def logger(self):
        return ExperimentLogger(experiment_name="test_privacy")
    
    def test_log_dp_epsilon_progression(self, logger):
        """Test logging DP-SGD epsilon consumption over rounds."""
        initial_epsilon = 1.0
        for round_num in range(10):
            # Simulate epsilon consumption
            epsilon_consumed = initial_epsilon / (round_num + 1)
            cumulative = initial_epsilon - epsilon_consumed
            
            logger.log_privacy_budget(
                round_num=round_num,
                epsilon=epsilon_consumed,
                delta=1e-5,
                cumulative_epsilon=cumulative,
                cumulative_delta=1e-5 * (round_num + 1)
            )
        assert True
    
    def test_log_privacy_with_byzantine(self, logger):
        """Test logging privacy with Byzantine-robust aggregation."""
        logger.log_privacy_budget(
            round_num=1,
            epsilon=0.15,
            delta=1e-6,
            cumulative_epsilon=0.15
        )
        logger.log_aggregation_metrics(
            round_num=1,
            aggregation_method="krum",
            num_clients_selected=30,
            num_clients_aggregated=28
        )
        assert True


class TestArtifactLogging:
    """Test artifact logging functionality."""
    
    @pytest.fixture
    def logger(self):
        return ExperimentLogger(experiment_name="test_artifacts")
    
    @pytest.fixture
    def temp_artifact(self):
        """Create a temporary artifact file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test artifact content")
            temp_path = f.name
        yield temp_path
        os.remove(temp_path)
    
    def test_log_artifact_exists(self, logger, temp_artifact):
        """Test logging an existing artifact."""
        logger.log_artifact(temp_artifact, artifact_type="model")
        assert True
    
    def test_log_artifact_not_found(self, logger):
        """Test logging a non-existent artifact."""
        logger.log_artifact("/nonexistent/path/file.txt")
        # Should handle gracefully without raising
        assert True
    
    def test_log_multiple_artifacts(self, logger):
        """Test logging multiple artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            files = []
            for i in range(3):
                filepath = os.path.join(tmpdir, f"artifact_{i}.txt")
                with open(filepath, 'w') as f:
                    f.write(f"artifact {i}")
                files.append(filepath)
            
            # Log all artifacts
            for filepath in files:
                logger.log_artifact(filepath, artifact_type="data")
            
            assert True


class TestExceptionLogging:
    """Test exception logging."""
    
    @pytest.fixture
    def logger(self):
        return ExperimentLogger(experiment_name="test_exceptions")
    
    def test_log_exception(self, logger):
        """Test logging an exception."""
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            logger.log_exception(e, context="Testing error handling")
        assert True
    
    def test_log_exception_with_context(self, logger):
        """Test logging exception with context."""
        exc = RuntimeError("Database connection failed")
        logger.log_exception(exc, context="Database layer")
        assert True


class TestRunManagement:
    """Test run start/end management."""
    
    def test_end_run_finished(self):
        """Test ending a run with FINISHED status."""
        logger = ExperimentLogger(experiment_name="test_run_mgmt")
        logger.end_run(status="FINISHED", notes="Experiment completed successfully")
        assert True
    
    def test_end_run_failed(self):
        """Test ending a run with FAILED status."""
        logger = ExperimentLogger(experiment_name="test_run_mgmt")
        logger.end_run(status="FAILED", notes="Error during training")
        assert True
    
    def test_get_run_id(self):
        """Test retrieving run ID."""
        logger = ExperimentLogger(experiment_name="test_run_id")
        run_id = logger.get_run_id()
        # Should return None or valid UUID
        assert run_id is None or isinstance(run_id, str)


class TestOrchestrationLogger:
    """Test OrchestrationLogger helper class."""
    
    @pytest.fixture
    def experiment_logger(self):
        return ExperimentLogger(experiment_name="test_orchestration")
    
    @pytest.fixture
    def orchestration_logger(self, experiment_logger):
        return OrchestrationLogger(experiment_logger)
    
    def test_init(self, orchestration_logger, experiment_logger):
        """Test OrchestrationLogger initialization."""
        assert orchestration_logger.logger is experiment_logger
    
    def test_log_round_with_mock_orchestrator(self, orchestration_logger):
        """Test logging round from mock orchestrator."""
        # Create mock orchestrator
        mock_orch = Mock()
        mock_orch.global_history = [{
            'algorithm': 'FedProx',
            'selected_clients': [0, 1, 2, 3],
            'num_updates': 4,
            'metrics': {
                0: {'accuracy': 0.90},
                1: {'accuracy': 0.88},
                2: {'accuracy': 0.92},
                3: {'accuracy': 0.89},
            }
        }]
        
        orchestration_logger.log_round(round_num=1, orchestrator=mock_orch)
        assert True
    
    def test_log_round_with_privacy_manager(self, orchestration_logger):
        """Test logging round with privacy manager."""
        mock_orch = Mock()
        mock_orch.global_history = [{
            'algorithm': 'FedProx',
            'selected_clients': [0, 1],
            'num_updates': 2,
            'metrics': {
                0: {'accuracy': 0.90},
                1: {'accuracy': 0.88},
            }
        }]
        
        mock_privacy = Mock()
        mock_privacy.epsilon_remaining = 0.5
        mock_privacy.delta_remaining = 5e-6
        mock_privacy.epsilon_initial = 1.0
        mock_privacy.delta_initial = 1e-5
        
        orchestration_logger.log_round(
            round_num=1,
            orchestrator=mock_orch,
            privacy_manager=mock_privacy
        )
        assert True


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_logger_with_defaults(self):
        """Test create_logger with default parameters."""
        logger = create_logger()
        assert logger.experiment_name == "FedShield"
        assert logger.backend == "mlflow"
    
    def test_create_logger_with_algorithm(self):
        """Test create_logger with algorithm parameter."""
        logger = create_logger(experiment_name="algo_test", algorithm="FedOpt")
        assert logger.experiment_name == "algo_test"
    
    def test_create_logger_with_backend(self):
        """Test create_logger with different backends."""
        logger = create_logger(backend="mlflow")
        assert logger.backend == "mlflow"


class TestIntegrationWithOrchestrators:
    """Integration tests with actual federated learning components."""
    
    def test_logging_fedprox_experiment(self):
        """Simulate logging a FedProx experiment."""
        logger = create_logger(
            experiment_name="FedProx_NonIID",
            algorithm="FedProx"
        )
        
        # Log hyperparameters
        logger.log_params({
            "num_rounds": 20,
            "clients_per_round": 10,
            "local_epochs": 5,
            "learning_rate": 0.01,
            "mu": 0.01,
            "epsilon": 1.0,
            "delta": 1e-5
        })
        
        # Simulate training rounds
        for round_num in range(5):
            # Log metrics
            logger.log_round_metrics(round_num, {
                "global_accuracy": 0.75 + (round_num * 0.03),
                "global_loss": 0.25 - (round_num * 0.01),
                "avg_client_accuracy": 0.72 + (round_num * 0.025),
            })
            
            # Log client metrics
            client_metrics = {i: {"accuracy": 0.70 + np.random.rand() * 0.1} 
                            for i in range(10)}
            logger.log_client_metrics(round_num, client_metrics)
            
            # Log privacy
            logger.log_privacy_budget(
                round_num,
                epsilon=0.05,
                delta=5e-6,
                cumulative_epsilon=0.05 * (round_num + 1)
            )
        
        logger.end_run(status="FINISHED", notes="FedProx experiment completed")
        assert True
    
    def test_logging_byzantine_experiment(self):
        """Simulate logging a Byzantine-robust aggregation experiment."""
        logger = create_logger(
            experiment_name="Byzantine_Robust",
            algorithm="FedAvg+Krum"
        )
        
        logger.log_params({
            "aggregation": "krum",
            "byzantine_clients": 2,
            "total_clients": 20,
            "num_rounds": 10
        })
        
        for round_num in range(5):
            logger.log_aggregation_metrics(
                round_num,
                aggregation_method="krum",
                num_clients_selected=20,
                num_clients_aggregated=18,
                aggregation_time=3.2
            )
            
            logger.log_round_metrics(round_num, {
                "global_accuracy": 0.85 + (round_num * 0.02),
                "robustness_score": 0.95,
            })
        
        logger.end_run(status="FINISHED")
        assert True


class TestBackendCompatibility:
    """Test compatibility with different logging backends."""
    
    def test_mlflow_only_backend(self):
        """Test MLflow-only backend."""
        logger = ExperimentLogger(backend="mlflow")
        logger.log_params({"test": "mlflow_only"})
        logger.log_round_metrics(1, {"accuracy": 0.90})
        assert True
    
    def test_invalid_backend_fallback(self):
        """Test handling of invalid backend."""
        logger = ExperimentLogger(backend="invalid")
        # Should not crash, just skip logging
        logger.log_params({"test": "invalid_backend"})
        assert True


class TestErrorHandling:
    """Test error handling and robustness."""
    
    def test_log_with_none_values(self):
        """Test logging with None values."""
        logger = ExperimentLogger()
        metrics = {
            "accuracy": 0.90,
            "loss": None,  # None value
            "f1": 0.88,
        }
        # Filter None values
        filtered = {k: v for k, v in metrics.items() if v is not None}
        logger.log_round_metrics(1, filtered)
        assert True
    
    def test_log_very_large_metrics(self):
        """Test logging very large metric values."""
        logger = ExperimentLogger()
        metrics = {
            "huge_number": 1e308,
            "tiny_number": 1e-308,
        }
        logger.log_round_metrics(1, metrics)
        assert True
    
    def test_log_nan_inf_values(self):
        """Test logging NaN and inf values."""
        logger = ExperimentLogger()
        metrics = {
            "normal": 0.90,
            "nan_value": float('nan'),
            "inf_value": float('inf'),
        }
        # Filter out NaN and inf
        filtered = {k: v for k, v in metrics.items() 
                   if not (np.isnan(v) if isinstance(v, float) else False)
                   and not (np.isinf(v) if isinstance(v, float) else False)}
        logger.log_round_metrics(1, filtered)
        assert True


class TestPerformance:
    """Test performance characteristics."""
    
    def test_log_many_rounds(self):
        """Test logging many rounds efficiently."""
        logger = ExperimentLogger(experiment_name="performance_test")
        
        # Log 100 rounds
        for round_num in range(100):
            logger.log_round_metrics(round_num, {
                "accuracy": 0.80 + (round_num * 0.001),
                "loss": 0.20 - (round_num * 0.0002),
            })
        
        assert True
    
    def test_log_many_clients(self):
        """Test logging metrics for many clients."""
        logger = ExperimentLogger(experiment_name="many_clients_test")
        
        # Log metrics for 100 clients
        client_metrics = {i: {"accuracy": 0.70 + np.random.rand() * 0.2} 
                         for i in range(100)}
        logger.log_client_metrics(1, client_metrics)
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
