"""
Byzantine-Robust Aggregation for Federated Learning

Implements robust aggregation methods that tolerate malicious/faulty clients:
- Krum selector: Selects the update with minimum sum of distances to others
- Median aggregation: Element-wise median (robust to outliers)
- Trimmed-mean aggregation: Remove extreme values, then average
- Anomaly detection: Score updates based on deviation from others

Theory:
    For n clients with up to f Byzantine faulty clients (f < n/3):
    These methods guarantee convergence even when f clients send adversarial updates.
    
    Key property: A single update cannot shift the aggregate significantly.
    
References:
    - Blanchard et al. "Machine Learning with Adversaries" (NIPS 2017)
    - Yin et al. "Byzantine-Robust Distributed Learning" (ICML 2018)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class AnomalyScore:
    """Score for detecting anomalous updates."""
    client_id: int
    score: float  # Higher = more anomalous
    is_outlier: bool
    std_devs_from_mean: float  # How many standard deviations away


class ByzantineAggregator:
    """
    Robust aggregation for Byzantine-tolerant federated learning.
    
    Provides multiple aggregation strategies that are robust to:
    - Malicious clients sending adversarial updates
    - Faulty clients sending corrupted data
    - Byzantine attacks (coordinated or independent)
    
    Guarantees: For f < n/3 Byzantine clients, convergence is guaranteed.
    """
    
    def __init__(
        self,
        num_clients: int,
        max_faulty: Optional[int] = None,
        method: str = "krum",
        anomaly_threshold: float = 3.0,
    ):
        """
        Initialize Byzantine aggregator.
        
        Args:
            num_clients: Total number of clients
            max_faulty: Maximum expected faulty clients (default: (n-1)/3)
            method: Default aggregation method ("krum", "median", "trimmed_mean")
            anomaly_threshold: Std devs above mean to flag as anomaly
        """
        self.num_clients = num_clients
        self.max_faulty = max_faulty or (num_clients - 1) // 3
        self.method = method
        self.anomaly_threshold = anomaly_threshold
        
        logger.info(
            f"ByzantineAggregator initialized: "
            f"clients={num_clients}, max_faulty={self.max_faulty}, method={method}"
        )
    
    def aggregate(
        self,
        client_updates: Dict[int, Dict],
        method: Optional[str] = None,
    ) -> Dict:
        """
        Aggregate updates using specified Byzantine-robust method.
        
        Args:
            client_updates: Dict of {client_id: update_dict}
            method: Aggregation method (uses default if not specified)
            
        Returns:
            Aggregated model update
        """
        method = method or self.method
        
        if method == "krum":
            return self.krum_selector(client_updates)
        elif method == "median":
            return self.median_aggregation(client_updates)
        elif method == "trimmed_mean":
            return self.trimmed_mean_aggregation(client_updates)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def krum_selector(
        self,
        client_updates: Dict[int, Dict],
        to_exclude: int = None,
    ) -> Dict:
        """
        Krum-select the most representative update.
        
        Selects the update with the smallest sum of distances to other updates.
        This is robust because a Byzantine update would have large distances
        to many others (or Byzantine updates wouldn't cluster together).
        
        Args:
            client_updates: Dict of {client_id: update_dict}
            to_exclude: Number of updates to exclude (default: max_faulty)
            
        Returns:
            Aggregated update (average of top-k selected updates)
            
        Theory:
            For each update u:
            - Compute L2 distance to all other updates
            - Distance(u, v) = ||u - v||_2
            - Krum score = sum of k nearest neighbors' distances
            - Select update with minimum Krum score
            
            With f Byzantine updates, non-Byzantine updates cluster together
            and have small scores. Byzantine updates are far from clusters
            and have large scores.
        """
        to_exclude = to_exclude or self.max_faulty
        
        if len(client_updates) == 1:
            return next(iter(client_updates.values()))
        
        # Flatten all updates for distance computation
        flat_updates = {}
        for cid, update in client_updates.items():
            flat = self._flatten_update(update)
            flat_updates[cid] = flat
        
        client_ids = list(flat_updates.keys())
        krum_scores = {}
        
        # Compute Krum score for each update
        for i, cid_i in enumerate(client_ids):
            u_i = flat_updates[cid_i]
            
            # Compute distances to all other updates
            distances = []
            for j, cid_j in enumerate(client_ids):
                if i != j:
                    u_j = flat_updates[cid_j]
                    dist = np.linalg.norm(u_i - u_j, ord=2)
                    distances.append(dist)
            
            # Krum score = sum of (n - f - 2) smallest distances
            num_to_sum = len(client_ids) - to_exclude - 2
            distances.sort()
            krum_scores[cid_i] = np.sum(distances[:max(1, num_to_sum)])
            
            logger.debug(f"Client {cid_i}: Krum score = {krum_scores[cid_i]:.4f}")
        
        # Select update(s) with minimum Krum score
        sorted_clients = sorted(krum_scores.items(), key=lambda x: x[1])
        selected_clients = [cid for cid, score in sorted_clients[:1]]  # Top-1
        
        logger.info(
            f"Krum selector: Selected client {selected_clients[0]} "
            f"(score={krum_scores[selected_clients[0]]:.4f})"
        )
        
        # Return the selected update
        return client_updates[selected_clients[0]]
    
    def median_aggregation(
        self,
        client_updates: Dict[int, Dict],
    ) -> Dict:
        """
        Element-wise median aggregation.
        
        Computes median of each parameter across clients.
        Robust because Byzantine values can't pull median far from honest clients.
        
        Args:
            client_updates: Dict of {client_id: update_dict}
            
        Returns:
            Aggregated update with median values
            
        Theory:
            For parameter p with updates p_1, p_2, ..., p_n:
            - Aggregated p = median(p_1, ..., p_n)
            - With f < n/2 Byzantine clients:
              - At least (n-f) > f honest clients
              - Median is always among honest updates or very close
            - Note: More lenient than Krum (tolerates f < n/2, not n/3)
        """
        aggregated = {}
        
        # Get all updates as flat arrays
        flat_updates = {}
        for cid, update in client_updates.items():
            flat = self._flatten_update(update)
            flat_updates[cid] = flat
        
        # Stack all updates: shape = (num_clients, param_size)
        client_ids = list(flat_updates.keys())
        stacked = np.stack([flat_updates[cid] for cid in client_ids], axis=0)
        
        # Compute element-wise median
        # axis=0 means median across clients (dimension 0)
        median_update = np.median(stacked, axis=0)
        
        logger.info(
            f"Median aggregation: {len(client_ids)} updates aggregated"
        )
        
        # Reshape back to original structure
        aggregated = self._reshape_update(
            median_update,
            next(iter(client_updates.values()))
        )
        
        return aggregated
    
    def trimmed_mean_aggregation(
        self,
        client_updates: Dict[int, Dict],
        trim_ratio: float = 0.2,
    ) -> Dict:
        """
        Trimmed-mean aggregation (remove extremes, then average).
        
        Removes the most extreme values before averaging.
        More robust than standard mean, less aggressive than median.
        
        Args:
            client_updates: Dict of {client_id: update_dict}
            trim_ratio: Fraction of extreme values to remove (each side)
            
        Returns:
            Aggregated update
            
        Theory:
            For each parameter:
            1. Sort values across clients
            2. Remove top (trim_ratio * n) and bottom (trim_ratio * n) values
            3. Average the remaining values
            
            For f Byzantine clients and trim_ratio = f/n:
            - All Byzantine values are removed
            - Average is purely from honest clients
            - Converges faster than median (uses more data)
        """
        aggregated = {}
        
        # Get all updates as flat arrays
        flat_updates = {}
        for cid, update in client_updates.items():
            flat = self._flatten_update(update)
            flat_updates[cid] = flat
        
        # Stack all updates: shape = (num_clients, param_size)
        client_ids = list(flat_updates.keys())
        stacked = np.stack([flat_updates[cid] for cid in client_ids], axis=0)
        
        # Compute trimmed mean
        # scipy.stats.trim_mean does this, but we'll implement manually
        num_trim = int(len(client_ids) * trim_ratio)
        
        if num_trim > 0:
            # Sort along axis 0 (across clients) and trim both ends
            sorted_vals = np.sort(stacked, axis=0)
            trimmed = sorted_vals[num_trim:-num_trim, :]
            trimmed_mean = np.mean(trimmed, axis=0)
        else:
            trimmed_mean = np.mean(stacked, axis=0)
        
        logger.info(
            f"Trimmed-mean aggregation: {len(client_ids)} updates, "
            f"removed {num_trim} extreme values from each side"
        )
        
        # Reshape back to original structure
        aggregated = self._reshape_update(
            trimmed_mean,
            next(iter(client_updates.values()))
        )
        
        return aggregated
    
    def detect_anomalies(
        self,
        client_updates: Dict[int, Dict],
    ) -> Dict[int, AnomalyScore]:
        """
        Detect anomalous (potentially Byzantine) updates.
        
        Scores each update based on deviation from the median.
        
        Args:
            client_updates: Dict of {client_id: update_dict}
            
        Returns:
            Dict of {client_id: AnomalyScore}
        """
        anomaly_scores = {}
        
        # Get all updates as flat arrays
        flat_updates = {}
        for cid, update in client_updates.items():
            flat = self._flatten_update(update)
            flat_updates[cid] = flat
        
        client_ids = list(flat_updates.keys())
        
        if len(client_ids) < 2:
            # Can't detect anomalies with <2 clients
            return {}
        
        # Compute pairwise distances to detect outliers
        for i, cid_i in enumerate(client_ids):
            u_i = flat_updates[cid_i]
            
            # Compute distances to all other updates
            distances_to_others = []
            for j, cid_j in enumerate(client_ids):
                if i != j:
                    u_j = flat_updates[cid_j]
                    dist = np.linalg.norm(u_i - u_j, ord=2)
                    distances_to_others.append(dist)
            
            # Score: average distance to others
            mean_dist = np.mean(distances_to_others)
            std_dist = np.std(distances_to_others)
            
            # Normalize: how many std devs above global mean?
            all_dists = []
            for j in range(len(client_ids)):
                for k in range(j + 1, len(client_ids)):
                    u_j = flat_updates[client_ids[j]]
                    u_k = flat_updates[client_ids[k]]
                    dist = np.linalg.norm(u_j - u_k, ord=2)
                    all_dists.append(dist)
            
            global_mean = np.mean(all_dists)
            global_std = np.std(all_dists)
            
            z_score = (mean_dist - global_mean) / (global_std + 1e-10)
            is_outlier = z_score > self.anomaly_threshold
            
            anomaly_scores[cid_i] = AnomalyScore(
                client_id=cid_i,
                score=z_score,
                is_outlier=is_outlier,
                std_devs_from_mean=z_score,
            )
            
            status = "⚠️  OUTLIER" if is_outlier else "✓ OK"
            logger.info(
                f"Client {cid_i}: Anomaly score = {z_score:.2f} {status}"
            )
        
        return anomaly_scores
    
    def get_statistics(
        self,
        client_updates: Dict[int, Dict],
    ) -> Dict:
        """
        Get statistics about the client updates.
        
        Useful for understanding data quality and Byzantine activity.
        """
        flat_updates = {}
        for cid, update in client_updates.items():
            flat = self._flatten_update(update)
            flat_updates[cid] = flat
        
        stacked = np.stack(list(flat_updates.values()), axis=0)
        
        return {
            'num_clients': len(client_updates),
            'mean_norm': np.mean([np.linalg.norm(u) for u in flat_updates.values()]),
            'std_norm': np.std([np.linalg.norm(u) for u in flat_updates.values()]),
            'min_norm': np.min([np.linalg.norm(u) for u in flat_updates.values()]),
            'max_norm': np.max([np.linalg.norm(u) for u in flat_updates.values()]),
            'param_mean': np.mean(stacked),
            'param_std': np.std(stacked),
        }
    
    def _flatten_update(self, update: Dict) -> np.ndarray:
        """Flatten update dict to 1D array."""
        if isinstance(update, np.ndarray):
            return update.flatten()
        
        # Flatten dict values
        flats = []
        for key in sorted(update.keys()):
            val = update[key]
            if isinstance(val, np.ndarray):
                flats.append(val.flatten())
        
        return np.concatenate(flats) if flats else np.array([])
    
    def _reshape_update(self, flat: np.ndarray, template: Dict) -> Dict:
        """Reshape flat array back to original dict structure."""
        result = {}
        offset = 0
        
        for key in sorted(template.keys()):
            val = template[key]
            if isinstance(val, np.ndarray):
                size = val.size
                result[key] = flat[offset:offset + size].reshape(val.shape)
                offset += size
        
        return result


class ClientQuarantine:
    """
    Tracks and quarantines suspicious clients.
    
    Clients with persistent anomalies are gradually excluded from aggregation.
    """
    
    def __init__(self, threshold: int = 3, window_size: int = 5):
        """
        Initialize quarantine system.
        
        Args:
            threshold: Number of anomalies to trigger quarantine
            window_size: Number of recent rounds to check
        """
        self.threshold = threshold
        self.window_size = window_size
        self.anomaly_history = {}  # {client_id: [bool, ...]}
        self.quarantined = set()
    
    def update(self, anomaly_scores: Dict[int, AnomalyScore]) -> None:
        """Update with new anomaly scores."""
        for cid, score in anomaly_scores.items():
            if cid not in self.anomaly_history:
                self.anomaly_history[cid] = []
            
            self.anomaly_history[cid].append(score.is_outlier)
            
            # Keep only recent history
            if len(self.anomaly_history[cid]) > self.window_size:
                self.anomaly_history[cid].pop(0)
            
            # Check for quarantine
            recent = self.anomaly_history[cid][-self.window_size:]
            anomaly_count = sum(recent)
            
            if anomaly_count >= self.threshold:
                self.quarantined.add(cid)
                logger.warning(f"Client {cid} quarantined ({anomaly_count}/{len(recent)} anomalies)")
            else:
                self.quarantined.discard(cid)
    
    def is_quarantined(self, client_id: int) -> bool:
        """Check if client is quarantined."""
        return client_id in self.quarantined
    
    def get_active_clients(self, client_ids: List[int]) -> List[int]:
        """Filter out quarantined clients."""
        return [cid for cid in client_ids if not self.is_quarantined(cid)]
    
    def get_status(self) -> Dict:
        """Get quarantine status."""
        return {
            'quarantined_count': len(self.quarantined),
            'quarantined_clients': list(self.quarantined),
            'total_history': len(self.anomaly_history),
        }


# Utility functions for Byzantine-robust comparison

def compare_robustness(
    client_updates: Dict[int, Dict],
    methods: List[str] = None,
) -> Dict[str, Dict]:
    """
    Compare different aggregation methods on the same data.
    
    Useful for understanding behavior under different Byzantine conditions.
    """
    methods = methods or ["krum", "median", "trimmed_mean"]
    aggregator = ByzantineAggregator(num_clients=len(client_updates))
    
    results = {}
    for method in methods:
        aggregated = aggregator.aggregate(client_updates, method=method)
        stats = aggregator.get_statistics(client_updates)
        results[method] = {
            'aggregated': aggregated,
            'statistics': stats,
        }
    
    return results
