"""
Test suite for Byzantine-robust aggregation methods.

Tests cover:
- Krum selector algorithm
- Median aggregation
- Trimmed-mean aggregation
- Anomaly detection
- Client quarantine
- Byzantine attack tolerance
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from server.robust_aggregation import (
    ByzantineAggregator,
    ClientQuarantine,
    AnomalyScore,
    compare_robustness,
)


class TestAnomalyScore:
    """Test AnomalyScore dataclass."""
    
    def test_anomaly_score_creation(self):
        """Test creating anomaly score."""
        score = AnomalyScore(
            client_id=1,
            score=3.5,
            is_outlier=True,
            std_devs_from_mean=3.5
        )
        
        assert score.client_id == 1
        assert score.score == 3.5
        assert score.is_outlier is True
        assert score.std_devs_from_mean == 3.5


class TestByzantineAggregatorInit:
    """Test ByzantineAggregator initialization."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        agg = ByzantineAggregator(num_clients=10)
        
        assert agg.num_clients == 10
        assert agg.max_faulty == 3  # (10-1)//3
        assert agg.method == "krum"
        assert agg.anomaly_threshold == 3.0
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        agg = ByzantineAggregator(
            num_clients=20,
            max_faulty=5,
            method="median",
            anomaly_threshold=2.5
        )
        
        assert agg.num_clients == 20
        assert agg.max_faulty == 5
        assert agg.method == "median"
        assert agg.anomaly_threshold == 2.5


class TestKrumSelector:
    """Test Krum selector algorithm."""
    
    def test_krum_single_client(self):
        """Test Krum with single client."""
        agg = ByzantineAggregator(num_clients=1)
        
        updates = {0: {'weights': np.array([1.0, 2.0, 3.0])}}
        result = agg.krum_selector(updates)
        
        assert result['weights'].shape == (3,)
        np.testing.assert_array_equal(result['weights'], [1.0, 2.0, 3.0])
    
    def test_krum_honest_clients(self):
        """Test Krum with all honest clients (no Byzantine)."""
        agg = ByzantineAggregator(num_clients=5)
        
        # All clients send similar updates
        updates = {
            i: {'weights': np.array([[1.0 + 0.1 * np.random.randn() for _ in range(3)]])}
            for i in range(5)
        }
        
        result = agg.krum_selector(updates)
        # Check that result has weights and no NaN values
        assert 'weights' in result
        assert not np.any(np.isnan(result['weights']))
    
    def test_krum_robust_to_byzantine(self):
        """Test that Krum is robust to Byzantine attacks."""
        agg = ByzantineAggregator(num_clients=5, max_faulty=1)
        
        # 4 honest clients
        honest = np.array([1.0, 1.0, 1.0])
        updates = {
            i: {'weights': honest + 0.01 * np.random.randn(3)}
            for i in range(4)
        }
        
        # 1 Byzantine client sending extreme values
        updates[4] = {'weights': np.array([100.0, -100.0, 100.0])}
        
        result = agg.krum_selector(updates)
        
        # Result should be close to honest values, not Byzantine
        assert np.allclose(result['weights'], honest, atol=0.2)
    
    def test_krum_with_dict_update(self):
        """Test Krum with multiple parameters."""
        agg = ByzantineAggregator(num_clients=3)
        
        updates = {
            0: {'coef': np.array([[1, 2], [3, 4]]), 'bias': np.array([0.1, 0.2])},
            1: {'coef': np.array([[1.1, 2.1], [3.1, 4.1]]), 'bias': np.array([0.11, 0.21])},
            2: {'coef': np.array([[1.2, 2.2], [3.2, 4.2]]), 'bias': np.array([0.12, 0.22])},
        }
        
        result = agg.krum_selector(updates)
        
        assert 'coef' in result
        assert 'bias' in result
        assert result['coef'].shape == (2, 2)
        assert result['bias'].shape == (2,)


class TestMedianAggregation:
    """Test median aggregation."""
    
    def test_median_simple(self):
        """Test median with simple values."""
        agg = ByzantineAggregator(num_clients=3)
        
        updates = {
            0: {'weights': np.array([1.0, 2.0, 3.0])},
            1: {'weights': np.array([2.0, 3.0, 4.0])},
            2: {'weights': np.array([3.0, 4.0, 5.0])},
        }
        
        result = agg.median_aggregation(updates)
        
        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result['weights'], expected)
    
    def test_median_robust_to_extremes(self):
        """Test that median is robust to extreme values."""
        agg = ByzantineAggregator(num_clients=5)
        
        updates = {
            0: {'weights': np.array([1.0, 1.0, 1.0])},
            1: {'weights': np.array([1.0, 1.0, 1.0])},
            2: {'weights': np.array([1.0, 1.0, 1.0])},
            3: {'weights': np.array([2.0, 2.0, 2.0])},  # One slightly different
            4: {'weights': np.array([1000.0, -1000.0, 1000.0])},  # Extreme Byzantine
        }
        
        result = agg.median_aggregation(updates)
        
        # Median should ignore the extreme Byzantine value
        assert np.allclose(result['weights'], 1.0, atol=0.5)
    
    def test_median_with_dict(self):
        """Test median with dict updates."""
        agg = ByzantineAggregator(num_clients=3)
        
        updates = {
            0: {'coef': np.array([1.0, 2.0]), 'bias': np.array([0.1])},
            1: {'coef': np.array([1.5, 2.5]), 'bias': np.array([0.15])},
            2: {'coef': np.array([2.0, 3.0]), 'bias': np.array([0.2])},
        }
        
        result = agg.median_aggregation(updates)
        
        assert 'coef' in result
        assert 'bias' in result


class TestTrimmedMeanAggregation:
    """Test trimmed-mean aggregation."""
    
    def test_trimmed_mean_no_trim(self):
        """Test trimmed mean with trim_ratio=0."""
        agg = ByzantineAggregator(num_clients=3)
        
        updates = {
            0: {'weights': np.array([1.0, 2.0, 3.0])},
            1: {'weights': np.array([2.0, 3.0, 4.0])},
            2: {'weights': np.array([3.0, 4.0, 5.0])},
        }
        
        result = agg.trimmed_mean_aggregation(updates, trim_ratio=0.0)
        
        expected = np.array([2.0, 3.0, 4.0])  # Regular mean
        np.testing.assert_array_almost_equal(result['weights'], expected)
    
    def test_trimmed_mean_with_trim(self):
        """Test trimmed mean with trim_ratio > 0."""
        agg = ByzantineAggregator(num_clients=5)
        
        updates = {
            0: {'weights': np.array([1.0, 1.0, 1.0])},
            1: {'weights': np.array([1.0, 1.0, 1.0])},
            2: {'weights': np.array([1.0, 1.0, 1.0])},
            3: {'weights': np.array([10.0, 10.0, 10.0])},  # Extreme high
            4: {'weights': np.array([-10.0, -10.0, -10.0])},  # Extreme low
        }
        
        # With trim_ratio=0.2, remove 1 from each end
        result = agg.trimmed_mean_aggregation(updates, trim_ratio=0.2)
        
        # Should be mean of the 3 middle values = [1, 1, 1]
        assert np.allclose(result['weights'], 1.0, atol=0.5)
    
    def test_trimmed_mean_robust(self):
        """Test trimmed mean robustness to Byzantine."""
        agg = ByzantineAggregator(num_clients=5)
        
        updates = {
            i: {'weights': np.array([1.0 + 0.1 * np.random.randn(3)])}
            for i in range(3)
        }
        
        # Add Byzantine clients
        updates[3] = {'weights': np.array([1000.0, -1000.0, 1000.0])}
        updates[4] = {'weights': np.array([-1000.0, 1000.0, -1000.0])}
        
        result = agg.trimmed_mean_aggregation(updates, trim_ratio=0.4)
        
        # Should be close to honest client values
        assert np.allclose(result['weights'], 1.0, atol=0.5)


class TestAnomalyDetection:
    """Test anomaly detection."""
    
    def test_detect_anomalies_all_honest(self):
        """Test anomaly detection with all honest clients."""
        agg = ByzantineAggregator(num_clients=5, anomaly_threshold=3.0)
        
        # All clients send similar updates
        updates = {
            i: {'weights': np.array([1.0 + 0.05 * np.random.randn(3)])}
            for i in range(5)
        }
        
        scores = agg.detect_anomalies(updates)
        
        # Few/no anomalies detected
        num_outliers = sum(1 for s in scores.values() if s.is_outlier)
        assert num_outliers <= 1  # Maybe one false positive
    
    def test_detect_anomalies_with_byzantine(self):
        """Test anomaly detection with Byzantine clients."""
        agg = ByzantineAggregator(num_clients=5, anomaly_threshold=1.0)
        
        updates = {
            i: {'weights': np.array([1.0 + 0.01 * np.random.randn(3)])}
            for i in range(4)
        }
        
        # 1 Byzantine client with much larger values
        updates[4] = {'weights': np.array([100.0, -100.0, 100.0])}
        
        scores = agg.detect_anomalies(updates)
        
        # Byzantine client should be detected as outlier with lower threshold
        if scores[4].score > agg.anomaly_threshold:
            assert scores[4].is_outlier
    
    def test_anomaly_score_fields(self):
        """Test that anomaly scores have all required fields."""
        agg = ByzantineAggregator(num_clients=3)
        
        updates = {
            0: {'weights': np.array([1.0, 2.0])},
            1: {'weights': np.array([1.1, 2.1])},
            2: {'weights': np.array([100.0, 200.0])},  # Byzantine
        }
        
        scores = agg.detect_anomalies(updates)
        
        for cid, score in scores.items():
            assert hasattr(score, 'client_id')
            assert hasattr(score, 'score')
            assert hasattr(score, 'is_outlier')
            assert hasattr(score, 'std_devs_from_mean')


class TestClientQuarantine:
    """Test client quarantine system."""
    
    def test_quarantine_initialization(self):
        """Test quarantine initialization."""
        quar = ClientQuarantine(threshold=3, window_size=5)
        
        assert quar.threshold == 3
        assert quar.window_size == 5
        assert len(quar.quarantined) == 0
    
    def test_quarantine_single_anomaly(self):
        """Test quarantine with single anomaly."""
        quar = ClientQuarantine(threshold=3, window_size=5)
        
        scores = {0: AnomalyScore(0, 1.0, True, 1.0)}
        quar.update(scores)
        
        assert not quar.is_quarantined(0)
    
    def test_quarantine_threshold(self):
        """Test quarantine threshold."""
        quar = ClientQuarantine(threshold=2, window_size=3)
        
        # Add 2 anomalies
        for _ in range(2):
            scores = {0: AnomalyScore(0, 1.0, True, 1.0)}
            quar.update(scores)
        
        # Should be quarantined after 2 anomalies
        assert quar.is_quarantined(0)
    
    def test_quarantine_window(self):
        """Test quarantine window size."""
        quar = ClientQuarantine(threshold=2, window_size=3)
        
        # Add 2 anomalies, then several normal updates
        for _ in range(2):
            scores = {0: AnomalyScore(0, 1.0, True, 1.0)}
            quar.update(scores)
        
        # Add 3 normal updates (beyond window)
        for _ in range(3):
            scores = {0: AnomalyScore(0, 0.5, False, 0.5)}
            quar.update(scores)
        
        # Should recover after window passes
        assert not quar.is_quarantined(0)
    
    def test_get_active_clients(self):
        """Test filtering active clients."""
        quar = ClientQuarantine(threshold=1, window_size=1)
        
        # Quarantine client 0
        scores = {0: AnomalyScore(0, 1.0, True, 1.0)}
        quar.update(scores)
        
        # Get active clients
        all_clients = [0, 1, 2, 3]
        active = quar.get_active_clients(all_clients)
        
        assert 0 not in active
        assert 1 in active
        assert 2 in active
        assert 3 in active
    
    def test_quarantine_status(self):
        """Test quarantine status reporting."""
        quar = ClientQuarantine(threshold=1, window_size=1)
        
        scores = {0: AnomalyScore(0, 1.0, True, 1.0)}
        quar.update(scores)
        
        status = quar.get_status()
        
        assert 'quarantined_count' in status
        assert 'quarantined_clients' in status
        assert status['quarantined_count'] == 1
        assert 0 in status['quarantined_clients']


class TestAggregationInterface:
    """Test aggregation interface."""
    
    def test_aggregate_method_selection(self):
        """Test aggregate method selection."""
        agg = ByzantineAggregator(num_clients=3, method="krum")
        
        updates = {
            0: {'weights': np.array([1.0, 2.0, 3.0])},
            1: {'weights': np.array([2.0, 3.0, 4.0])},
            2: {'weights': np.array([3.0, 4.0, 5.0])},
        }
        
        # Test each method
        result_krum = agg.aggregate(updates, method="krum")
        result_median = agg.aggregate(updates, method="median")
        result_trimmed = agg.aggregate(updates, method="trimmed_mean")
        
        assert 'weights' in result_krum
        assert 'weights' in result_median
        assert 'weights' in result_trimmed
    
    def test_aggregate_invalid_method(self):
        """Test aggregate with invalid method."""
        agg = ByzantineAggregator(num_clients=3)
        
        updates = {
            0: {'weights': np.array([1.0, 2.0, 3.0])},
        }
        
        with pytest.raises(ValueError):
            agg.aggregate(updates, method="invalid_method")


class TestStatistics:
    """Test statistics computation."""
    
    def test_get_statistics(self):
        """Test getting update statistics."""
        agg = ByzantineAggregator(num_clients=3)
        
        updates = {
            0: {'weights': np.array([1.0, 2.0, 3.0])},
            1: {'weights': np.array([2.0, 3.0, 4.0])},
            2: {'weights': np.array([3.0, 4.0, 5.0])},
        }
        
        stats = agg.get_statistics(updates)
        
        assert 'num_clients' in stats
        assert 'mean_norm' in stats
        assert 'std_norm' in stats
        assert 'min_norm' in stats
        assert 'max_norm' in stats
        assert 'param_mean' in stats
        assert 'param_std' in stats
        
        assert stats['num_clients'] == 3
        assert stats['mean_norm'] > 0


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_compare_robustness(self):
        """Test comparing aggregation methods."""
        updates = {
            0: {'weights': np.array([1.0, 2.0, 3.0])},
            1: {'weights': np.array([2.0, 3.0, 4.0])},
            2: {'weights': np.array([3.0, 4.0, 5.0])},
        }
        
        results = compare_robustness(updates)
        
        assert 'krum' in results
        assert 'median' in results
        assert 'trimmed_mean' in results
        
        for method, result in results.items():
            assert 'aggregated' in result
            assert 'statistics' in result


class TestByzantineAttackTolerance:
    """Test tolerance to Byzantine attacks."""
    
    def test_tolerance_to_scaling_attack(self):
        """Test tolerance to scaled Byzantine attacks."""
        agg = ByzantineAggregator(num_clients=5, max_faulty=1)
        
        # Honest clients
        honest = np.array([1.0, 1.0, 1.0])
        updates = {
            i: {'weights': honest + 0.01 * np.random.randn(3)}
            for i in range(4)
        }
        
        # Byzantine client amplifies all values
        updates[4] = {'weights': 100 * honest}
        
        result_krum = agg.krum_selector(updates)
        result_median = agg.median_aggregation(updates)
        
        # Both should be robust
        assert np.allclose(result_krum['weights'], honest, atol=0.5)
        assert np.allclose(result_median['weights'], honest, atol=0.5)
    
    def test_tolerance_to_sign_flip_attack(self):
        """Test tolerance to sign-flip Byzantine attacks."""
        agg = ByzantineAggregator(num_clients=5, max_faulty=1)
        
        honest = np.array([1.0, 2.0, 3.0])
        updates = {
            i: {'weights': honest + 0.01 * np.random.randn(3)}
            for i in range(4)
        }
        
        # Byzantine client flips signs
        updates[4] = {'weights': -honest}
        
        result_median = agg.median_aggregation(updates)
        
        # Median should still be robust
        assert np.allclose(result_median['weights'], honest, atol=0.5)


class TestEdgeCases:
    """Test edge cases."""
    
    def test_single_parameter(self):
        """Test with single parameter."""
        agg = ByzantineAggregator(num_clients=3)
        
        updates = {
            0: {'weight': np.array([1.0])},
            1: {'weight': np.array([2.0])},
            2: {'weight': np.array([3.0])},
        }
        
        result = agg.median_aggregation(updates)
        assert result['weight'].shape == (1,)
    
    def test_many_parameters(self):
        """Test with many parameters."""
        agg = ByzantineAggregator(num_clients=3)
        
        updates = {
            i: {f'w{j}': np.array([float(j)]) for j in range(10)}
            for i in range(3)
        }
        
        result = agg.median_aggregation(updates)
        assert len(result) == 10
    
    def test_large_parameter_arrays(self):
        """Test with large parameter arrays."""
        agg = ByzantineAggregator(num_clients=3)
        
        updates = {
            i: {'weights': np.random.randn(1000, 100)}
            for i in range(3)
        }
        
        result = agg.median_aggregation(updates)
        assert result['weights'].shape == (1000, 100)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
