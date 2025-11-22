"""
Experiment Logging Module for FedShield

Integrates MLflow and Weights & Biases for comprehensive experiment tracking.
Logs metrics, model versions, privacy budgets, accuracy curves, and hyperparameters.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import mlflow
from mlflow.entities import Metric, Param
import numpy as np

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """
    Unified experiment logger for FedShield federated learning.
    
    Supports MLflow and W&B backends for tracking:
    - Per-round metrics (accuracy, loss, convergence)
    - Client-level statistics
    - Privacy budget consumption (DP-SGD)
    - Model versions and hyperparameters
    - Communication costs and bandwidth usage
    """
    
    def __init__(self, 
                 experiment_name: str = "FedShield",
                 run_name: str = None,
                 tracking_uri: str = None,
                 backend: str = "mlflow",
                 log_dir: str = "./mlruns"):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            run_name: Name of this specific run (if None, auto-generated)
            tracking_uri: MLflow tracking URI (if None, uses local ./mlruns)
            backend: "mlflow", "wandb", or "both"
            log_dir: Directory for local MLflow artifacts
        """
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.backend = backend.lower()
        self.tracking_uri = tracking_uri
        self.log_dir = log_dir
        
        # Initialize MLflow
        if self.backend in ["mlflow", "both"]:
            self._init_mlflow()
        
        # Initialize W&B
        if self.backend in ["wandb", "both"]:
            self._init_wandb()
        
        self.metrics_buffer = {}  # Buffer for batching metric logs
        self.artifacts_dir = os.path.join(self.log_dir, "artifacts")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
        logger.info(f"ExperimentLogger initialized: {self.experiment_name}/{self.run_name}")
    
    def _init_mlflow(self):
        """Initialize MLflow tracking."""
        try:
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            else:
                mlflow.set_tracking_uri(f"file:{os.path.abspath(self.log_dir)}")
            
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run(run_name=self.run_name)
            logger.info(f"MLflow tracking initialized: {self.tracking_uri}")
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        try:
            import wandb
            wandb.init(
                project=self.experiment_name,
                name=self.run_name,
                config={}
            )
            logger.info("W&B tracking initialized")
        except Exception as e:
            logger.warning(f"W&B not available or not configured: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters and configuration."""
        try:
            if self.backend in ["mlflow", "both"]:
                for key, value in params.items():
                    # Convert complex types to string for MLflow
                    if isinstance(value, (dict, list)):
                        mlflow.log_param(key, json.dumps(value))
                    else:
                        mlflow.log_param(key, value)
            
            if self.backend in ["wandb", "both"]:
                try:
                    import wandb
                    wandb.config.update(params)
                except:
                    pass
            
            logger.debug(f"Logged {len(params)} parameters")
        except Exception as e:
            logger.error(f"Failed to log params: {e}")
    
    def log_round_metrics(self, round_num: int, metrics: Dict[str, float]):
        """
        Log per-round metrics.
        
        Typical metrics:
        - global_accuracy: Global model accuracy on validation set
        - global_loss: Global model loss
        - avg_client_accuracy: Average across all clients
        - communication_cost: Bytes transmitted this round
        - privacy_budget_consumed: Privacy budget used (DP-SGD)
        - convergence_rate: Change in accuracy from previous round
        """
        try:
            if self.backend in ["mlflow", "both"]:
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value, step=round_num)
            
            if self.backend in ["wandb", "both"]:
                try:
                    import wandb
                    log_dict = {f"round/{k}": v for k, v in metrics.items()}
                    log_dict["round"] = round_num
                    wandb.log(log_dict)
                except:
                    pass
            
            logger.debug(f"Logged metrics for round {round_num}: {list(metrics.keys())}")
        except Exception as e:
            logger.error(f"Failed to log round metrics: {e}")
    
    def log_client_metrics(self, round_num: int, client_metrics: Dict[int, Dict[str, float]]):
        """
        Log per-client metrics.
        
        Args:
            round_num: Round number
            client_metrics: Dict mapping client_id -> {accuracy, loss, samples, ...}
        """
        try:
            metrics_summary = {
                "num_clients": len(client_metrics),
                "avg_accuracy": np.mean([m.get('accuracy', 0) for m in client_metrics.values()]),
                "std_accuracy": np.std([m.get('accuracy', 0) for m in client_metrics.values()]),
                "min_accuracy": np.min([m.get('accuracy', 0) for m in client_metrics.values()]),
                "max_accuracy": np.max([m.get('accuracy', 0) for m in client_metrics.values()]),
            }
            
            if self.backend in ["mlflow", "both"]:
                for metric_name, metric_value in metrics_summary.items():
                    mlflow.log_metric(f"clients/{metric_name}", metric_value, step=round_num)
            
            if self.backend in ["wandb", "both"]:
                try:
                    import wandb
                    log_dict = {f"clients/{k}": v for k, v in metrics_summary.items()}
                    wandb.log(log_dict)
                except:
                    pass
            
            logger.debug(f"Logged client metrics summary for round {round_num}")
        except Exception as e:
            logger.error(f"Failed to log client metrics: {e}")
    
    def log_privacy_budget(self, round_num: int, epsilon: float, delta: float, 
                          cumulative_epsilon: float = None, cumulative_delta: float = None):
        """
        Log differential privacy budget consumption.
        
        Args:
            round_num: Round number
            epsilon: Epsilon consumed this round
            delta: Delta consumed this round
            cumulative_epsilon: Total epsilon consumed so far
            cumulative_delta: Total delta consumed so far
        """
        try:
            metrics = {
                "privacy/epsilon_round": epsilon,
                "privacy/delta_round": delta,
            }
            
            if cumulative_epsilon is not None:
                metrics["privacy/epsilon_cumulative"] = cumulative_epsilon
            if cumulative_delta is not None:
                metrics["privacy/delta_cumulative"] = cumulative_delta
            
            if self.backend in ["mlflow", "both"]:
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value, step=round_num)
            
            if self.backend in ["wandb", "both"]:
                try:
                    import wandb
                    wandb.log(metrics)
                except:
                    pass
            
            logger.debug(f"Logged privacy budget for round {round_num}")
        except Exception as e:
            logger.error(f"Failed to log privacy budget: {e}")
    
    def log_model_version(self, round_num: int, model_path: str = None, 
                         algorithm: str = None, metrics: Dict[str, float] = None):
        """
        Log model version checkpoint.
        
        Args:
            round_num: Round number
            model_path: Path to model file (optional)
            algorithm: Algorithm name (FedAvg, FedProx, FedOpt, etc.)
            metrics: Model performance metrics
        """
        try:
            tags = {
                "round": round_num,
                "algorithm": algorithm or "unknown",
            }
            
            if self.backend in ["mlflow", "both"]:
                mlflow.set_tags(tags)
                if metrics:
                    for key, value in metrics.items():
                        mlflow.log_metric(f"model/{key}", value, step=round_num)
            
            if self.backend in ["wandb", "both"]:
                try:
                    import wandb
                    wandb.run.tags = list(tags.values())
                except:
                    pass
            
            logger.debug(f"Logged model version for round {round_num}")
        except Exception as e:
            logger.error(f"Failed to log model version: {e}")
    
    def log_aggregation_metrics(self, round_num: int, aggregation_method: str,
                               num_clients_selected: int, num_clients_aggregated: int,
                               aggregation_time: float = None):
        """
        Log aggregation-specific metrics.
        
        Args:
            round_num: Round number
            aggregation_method: Method used (FedAvg, median, krum, trimmed-mean)
            num_clients_selected: Number of clients selected
            num_clients_aggregated: Number of clients actually aggregated
            aggregation_time: Time taken for aggregation (seconds)
        """
        try:
            metrics = {
                "aggregation/method": aggregation_method,
                "aggregation/clients_selected": num_clients_selected,
                "aggregation/clients_aggregated": num_clients_aggregated,
                "aggregation/participation_rate": num_clients_aggregated / num_clients_selected if num_clients_selected > 0 else 0,
            }
            
            if aggregation_time is not None:
                metrics["aggregation/time_seconds"] = aggregation_time
            
            if self.backend in ["mlflow", "both"]:
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value, step=round_num)
            
            if self.backend in ["wandb", "both"]:
                try:
                    import wandb
                    wandb.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
                except:
                    pass
            
            logger.debug(f"Logged aggregation metrics for round {round_num}")
        except Exception as e:
            logger.error(f"Failed to log aggregation metrics: {e}")
    
    def log_communication_stats(self, round_num: int, 
                               bytes_uploaded: int = None,
                               bytes_downloaded: int = None,
                               num_clients: int = None):
        """
        Log communication and bandwidth metrics.
        
        Args:
            round_num: Round number
            bytes_uploaded: Bytes uploaded to server
            bytes_downloaded: Bytes downloaded from server
            num_clients: Number of participating clients
        """
        try:
            metrics = {}
            
            if bytes_uploaded is not None:
                metrics["communication/bytes_uploaded"] = bytes_uploaded
                metrics["communication/mb_uploaded"] = bytes_uploaded / (1024 ** 2)
            
            if bytes_downloaded is not None:
                metrics["communication/bytes_downloaded"] = bytes_downloaded
                metrics["communication/mb_downloaded"] = bytes_downloaded / (1024 ** 2)
            
            if num_clients is not None and bytes_uploaded is not None:
                metrics["communication/avg_bytes_per_client"] = bytes_uploaded / num_clients
            
            if self.backend in ["mlflow", "both"]:
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value, step=round_num)
            
            if self.backend in ["wandb", "both"]:
                try:
                    import wandb
                    wandb.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
                except:
                    pass
            
            logger.debug(f"Logged communication stats for round {round_num}")
        except Exception as e:
            logger.error(f"Failed to log communication stats: {e}")
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "model"):
        """
        Log artifact (model, plot, data, etc.).
        
        Args:
            artifact_path: Path to artifact file
            artifact_type: Type of artifact (model, plot, data, config)
        """
        try:
            if not os.path.exists(artifact_path):
                logger.warning(f"Artifact not found: {artifact_path}")
                return
            
            if self.backend in ["mlflow", "both"]:
                mlflow.log_artifact(artifact_path, artifact_path=artifact_type)
            
            logger.debug(f"Logged artifact: {artifact_path}")
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")
    
    def log_exception(self, exception: Exception, context: str = None):
        """Log exception details."""
        try:
            error_msg = f"{context}: {str(exception)}" if context else str(exception)
            
            if self.backend in ["mlflow", "both"]:
                mlflow.log_param("error", error_msg)
            
            if self.backend in ["wandb", "both"]:
                try:
                    import wandb
                    wandb.log({"error": error_msg})
                except:
                    pass
            
            logger.error(f"Logged exception: {error_msg}")
        except Exception as e:
            logger.error(f"Failed to log exception: {e}")
    
    def end_run(self, status: str = "FINISHED", notes: str = None):
        """
        End the current experiment run.
        
        Args:
            status: "FINISHED", "FAILED", "KILLED"
            notes: Optional notes about the run
        """
        try:
            if notes:
                if self.backend in ["mlflow", "both"]:
                    mlflow.log_param("notes", notes)
            
            if self.backend in ["mlflow", "both"]:
                mlflow.end_run(status=status)
            
            if self.backend in ["wandb", "both"]:
                try:
                    import wandb
                    wandb.finish()
                except:
                    pass
            
            logger.info(f"Experiment run ended with status: {status}")
        except Exception as e:
            logger.error(f"Failed to end run: {e}")
    
    def get_run_id(self) -> str:
        """Get the current MLflow run ID."""
        try:
            if self.backend in ["mlflow", "both"]:
                return mlflow.active_run().info.run_id
        except:
            pass
        return None


class OrchestrationLogger:
    """
    Helper class to integrate ExperimentLogger with Orchestrator classes.
    
    Usage:
        logger = ExperimentLogger("FedShield", "fedprox_experiment")
        orch_logger = OrchestrationLogger(logger)
        
        # In your orchestrator loop:
        orchestrator.simulate_round(...)
        orch_logger.log_round(round_num, orchestrator)
    """
    
    def __init__(self, experiment_logger: ExperimentLogger):
        """Initialize orchestration logger."""
        self.logger = experiment_logger
    
    def log_round(self, round_num: int, orchestrator: Any, 
                 privacy_manager: Optional[Any] = None):
        """
        Log metrics from an orchestrator round.
        
        Args:
            round_num: Round number
            orchestrator: FedAvgOrchestrator, FedProxOrchestrator, or FedOptOrchestrator
            privacy_manager: PrivacyManager instance (if using DP-SGD)
        """
        try:
            # Extract metrics from orchestrator history
            if hasattr(orchestrator, 'global_history') and orchestrator.global_history:
                last_round = orchestrator.global_history[-1]
                
                # Log round metrics
                round_metrics = {
                    "algorithm": last_round.get('algorithm', 'unknown'),
                    "num_selected_clients": len(last_round.get('selected_clients', [])),
                    "num_updates": last_round.get('num_updates', 0),
                }
                self.logger.log_round_metrics(round_num, round_metrics)
                
                # Log client metrics
                if 'metrics' in last_round:
                    self.logger.log_client_metrics(round_num, last_round['metrics'])
                
                # Log aggregation info
                aggregation_method = last_round.get('algorithm', 'FedAvg')
                self.logger.log_aggregation_metrics(
                    round_num,
                    aggregation_method,
                    len(last_round.get('selected_clients', [])),
                    last_round.get('num_updates', 0)
                )
            
            # Log privacy budget if available
            if privacy_manager and hasattr(privacy_manager, 'epsilon'):
                self.logger.log_privacy_budget(
                    round_num,
                    privacy_manager.epsilon_remaining,
                    privacy_manager.delta_remaining,
                    cumulative_epsilon=privacy_manager.epsilon_initial - privacy_manager.epsilon_remaining,
                    cumulative_delta=privacy_manager.delta_initial - privacy_manager.delta_remaining
                )
        
        except Exception as e:
            logger.error(f"Failed to log round {round_num}: {e}")


# Convenience function for quick setup
def create_logger(experiment_name: str = "FedShield",
                 algorithm: str = None,
                 backend: str = "mlflow",
                 **kwargs) -> ExperimentLogger:
    """
    Convenience function to create and configure an experiment logger.
    
    Args:
        experiment_name: Name of experiment
        algorithm: Algorithm name (FedAvg, FedProx, FedOpt, etc.)
        backend: "mlflow", "wandb", or "both"
        **kwargs: Additional arguments for ExperimentLogger
    
    Returns:
        Configured ExperimentLogger instance
    """
    logger_instance = ExperimentLogger(
        experiment_name=experiment_name,
        backend=backend,
        **kwargs
    )
    
    if algorithm:
        logger_instance.log_params({"algorithm": algorithm})
    
    return logger_instance
