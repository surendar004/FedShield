"""
Privacy Manager for Differential Privacy in Federated Learning

Implements DP-SGD (Differentially Private Stochastic Gradient Descent) with:
- L2 gradient clipping
- Gaussian noise injection
- Privacy budget tracking (ε-δ differential privacy)
- Per-round privacy accounting

Theory:
    For ε-δ differential privacy guarantee across T rounds:
    - Noise scale: σ = sqrt(2 * ln(1.25/δ)) / ε  
    - Privacy budget per round: ε_round = ε / sqrt(T)
    - Cumulative privacy: (ε, δ)-DP after T rounds

References:
    - Abadi et al. "Deep Learning with Differential Privacy" (ICML 2016)
    - Kairouz et al. "Differentially Private Federated Learning" (AISTATS 2021)
"""

import numpy as np
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class PrivacyParams:
    """Configuration for differential privacy."""
    epsilon: float = 1.0  # Privacy budget
    delta: float = 1e-5   # Failure probability
    clip_norm: float = 1.0  # L2 clipping threshold
    

class PrivacyBudget:
    """Tracks accumulated privacy loss across training rounds."""
    
    def __init__(self, epsilon: float, delta: float, num_rounds: int):
        """
        Initialize privacy budget.
        
        Args:
            epsilon: Total privacy budget for entire training
            delta: Failure probability
            num_rounds: Expected number of training rounds
        """
        self.total_epsilon = epsilon
        self.total_delta = delta
        self.num_rounds = num_rounds
        self.rounds_completed = 0
        
        # Per-round budget (using advanced composition theorem)
        # For (ε, δ)-DP across T rounds, allocate ε_round = ε / sqrt(T)
        self.epsilon_per_round = epsilon / np.sqrt(num_rounds)
        
        logger.info(
            f"Privacy Budget Initialized - "
            f"Total ε={epsilon}, δ={delta}, Rounds={num_rounds}, "
            f"Per-round ε={self.epsilon_per_round:.4f}"
        )
    
    def use_round(self):
        """Account for one round of training."""
        self.rounds_completed += 1
        remaining_epsilon = self.epsilon_remaining()
        
        logger.debug(
            f"Round {self.rounds_completed}/{self.num_rounds} - "
            f"Remaining ε={remaining_epsilon:.4f}"
        )
        
        return remaining_epsilon
    
    def epsilon_remaining(self) -> float:
        """Get remaining privacy budget (in ε units)."""
        return self.total_epsilon - (self.rounds_completed * self.epsilon_per_round)
    
    def get_status(self) -> Dict[str, Union[float, int]]:
        """Get privacy budget status."""
        return {
            'epsilon_total': self.total_epsilon,
            'epsilon_per_round': self.epsilon_per_round,
            'epsilon_used': self.rounds_completed * self.epsilon_per_round,
            'epsilon_remaining': self.epsilon_remaining(),
            'delta': self.total_delta,
            'rounds_completed': self.rounds_completed,
            'rounds_total': self.num_rounds,
        }


class PrivacyManager:
    """
    Manages differential privacy operations for federated learning.
    
    Provides:
    - L2 norm gradient clipping
    - Gaussian noise injection
    - Privacy budget tracking
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0,
        num_rounds: int = 10,
    ):
        """
        Initialize PrivacyManager.
        
        Args:
            epsilon: Privacy budget (smaller = more private = noisier)
            delta: Failure probability (typically 1/n where n = num_clients)
            clip_norm: L2 clipping threshold for gradients
            num_rounds: Total number of training rounds
            
        Theory:
            (ε, δ)-DP achieved by:
            1. Clipping gradients to bounded L2 norm
            2. Adding Gaussian noise N(0, σ²) where σ = sqrt(2*ln(1.25/δ)) / ε_round
            3. Using advanced composition theorem to track cumulative privacy
        """
        self.params = PrivacyParams(
            epsilon=epsilon,
            delta=delta,
            clip_norm=clip_norm,
        )
        
        self.privacy_budget = PrivacyBudget(epsilon, delta, num_rounds)
        
        # Calculate noise standard deviation for Gaussian mechanism
        # Formula: σ = sqrt(2 * ln(1.25/δ)) / ε_per_round
        # Derivation: From concentrated differential privacy theorem
        self.noise_multiplier = np.sqrt(2.0 * np.log(1.25 / delta))
        
        logger.info(
            f"PrivacyManager initialized: "
            f"ε={epsilon}, δ={delta}, clip_norm={clip_norm}, "
            f"noise_multiplier={self.noise_multiplier:.4f}"
        )
    
    def clip_gradients(self, gradients: Union[Dict, np.ndarray]) -> Union[Dict, np.ndarray]:
        """
        Apply L2 norm clipping to gradients.
        
        Clips gradients to have L2 norm ≤ clip_norm. This bounds the 
        sensitivity of the aggregation function.
        
        Args:
            gradients: Dict of parameter gradients or flat array
            
        Returns:
            Clipped gradients with same structure as input
            
        Theory:
            For gradient g:
            - Compute L2 norm: ||g||_2 = sqrt(sum(g_i²))
            - Clipping factor: min(1, clip_norm / ||g||_2)
            - Result: g_clipped = g * clipping_factor
            
            This ensures ||g_clipped||_2 ≤ clip_norm
        """
        if isinstance(gradients, dict):
            return self._clip_gradients_dict(gradients)
        else:
            return self._clip_gradients_array(gradients)
    
    def _clip_gradients_dict(self, gradients: Dict) -> Dict:
        """Clip gradients in dictionary format."""
        # Flatten all gradients to compute total L2 norm
        flat_grads = []
        for key, value in gradients.items():
            if isinstance(value, np.ndarray):
                flat_grads.append(value.flatten())
        
        if not flat_grads:
            return gradients
        
        # Compute L2 norm
        all_grads = np.concatenate(flat_grads)
        grad_norm = np.linalg.norm(all_grads, ord=2)
        
        # Compute clipping factor
        clipping_factor = min(1.0, self.params.clip_norm / (grad_norm + 1e-10))
        
        # Apply clipping
        clipped = {}
        for key, value in gradients.items():
            if isinstance(value, np.ndarray):
                clipped[key] = value * clipping_factor
            else:
                clipped[key] = value
        
        # Log clipping statistics
        if grad_norm > self.params.clip_norm:
            logger.debug(
                f"Gradient clipping: norm={grad_norm:.4f} → "
                f"factor={clipping_factor:.4f}"
            )
        
        return clipped
    
    def _clip_gradients_array(self, gradients: np.ndarray) -> np.ndarray:
        """Clip gradients in array format."""
        grad_norm = np.linalg.norm(gradients, ord=2)
        clipping_factor = min(1.0, self.params.clip_norm / (grad_norm + 1e-10))
        
        if grad_norm > self.params.clip_norm:
            logger.debug(
                f"Gradient clipping: norm={grad_norm:.4f} → "
                f"factor={clipping_factor:.4f}"
            )
        
        return gradients * clipping_factor
    
    def add_gaussian_noise(
        self,
        clipped_gradients: Union[Dict, np.ndarray],
    ) -> Union[Dict, np.ndarray]:
        """
        Add Gaussian noise to clipped gradients.
        
        Adds calibrated Gaussian noise to achieve differential privacy.
        
        Args:
            clipped_gradients: Clipped gradients (output of clip_gradients())
            
        Returns:
            Noisy gradients with same structure as input
            
        Theory:
            For ε-DP on sensitivity-bounded function f:
            - Noise scale: σ = sensitivity / ε
            - Add noise: result = f(x) + N(0, σ²I)
            
            For (ε, δ)-DP:
            - Noise scale: σ = sqrt(2 * ln(1.25/δ)) / ε
            - Each coordinate gets independent Gaussian noise
        """
        if isinstance(clipped_gradients, dict):
            return self._add_noise_dict(clipped_gradients)
        else:
            return self._add_noise_array(clipped_gradients)
    
    def _add_noise_dict(self, clipped_gradients: Dict) -> Dict:
        """Add noise to gradients in dictionary format."""
        # Get privacy budget for this round
        epsilon_round = self.privacy_budget.epsilon_per_round
        
        # Calculate noise standard deviation
        # σ = sqrt(2 * ln(1.25/δ)) * clip_norm / ε_round
        sigma = self.noise_multiplier * self.params.clip_norm / (epsilon_round + 1e-10)
        
        noisy = {}
        for key, value in clipped_gradients.items():
            if isinstance(value, np.ndarray):
                # Add independent Gaussian noise to each element
                noise = np.random.normal(0, sigma, size=value.shape)
                noisy[key] = value + noise
            else:
                noisy[key] = value
        
        logger.debug(
            f"Gaussian noise added: σ={sigma:.4f}, "
            f"ε_round={epsilon_round:.4f}"
        )
        
        return noisy
    
    def _add_noise_array(self, clipped_gradients: np.ndarray) -> np.ndarray:
        """Add noise to gradients in array format."""
        epsilon_round = self.privacy_budget.epsilon_per_round
        sigma = self.noise_multiplier * self.params.clip_norm / (epsilon_round + 1e-10)
        
        noise = np.random.normal(0, sigma, size=clipped_gradients.shape)
        
        logger.debug(
            f"Gaussian noise added: σ={sigma:.4f}, "
            f"ε_round={epsilon_round:.4f}"
        )
        
        return clipped_gradients + noise
    
    def apply_privacy(
        self,
        gradients: Union[Dict, np.ndarray],
    ) -> Union[Dict, np.ndarray]:
        """
        Apply both clipping and noise in one step.
        
        Convenience method that applies the full DP-SGD pipeline:
        1. Clip gradients to bound sensitivity
        2. Add Gaussian noise for privacy
        
        Args:
            gradients: Raw gradients from local training
            
        Returns:
            Privacy-enhanced gradients
        """
        clipped = self.clip_gradients(gradients)
        noisy = self.add_gaussian_noise(clipped)
        self.privacy_budget.use_round()
        return noisy
    
    def get_privacy_status(self) -> Dict:
        """Get detailed privacy status."""
        status = self.privacy_budget.get_status()
        status.update({
            'clip_norm': self.params.clip_norm,
            'noise_multiplier': self.noise_multiplier,
        })
        return status
    
    def get_noise_scale(self) -> float:
        """
        Get the current noise standard deviation.
        
        Useful for monitoring the privacy-utility tradeoff.
        """
        epsilon_round = self.privacy_budget.epsilon_per_round
        return self.noise_multiplier * self.params.clip_norm / (epsilon_round + 1e-10)


class DPSGDTrainer:
    """
    Trainer that applies DP-SGD during model training.
    
    Wrapper around model training that applies privacy operations
    between gradient computation and parameter updates.
    """
    
    def __init__(self, privacy_manager: PrivacyManager):
        """Initialize DP-SGD trainer."""
        self.privacy_manager = privacy_manager
        self.stats = {
            'gradients_clipped': 0,
            'noise_added': 0,
            'training_steps': 0,
        }
    
    def clip_and_noise_gradients(
        self,
        gradients: Union[Dict, np.ndarray],
    ) -> Union[Dict, np.ndarray]:
        """
        Apply DP-SGD to gradients.
        
        Args:
            gradients: Gradients from backpropagation
            
        Returns:
            DP-enhanced gradients ready for parameter update
        """
        self.stats['gradients_clipped'] += 1
        self.stats['noise_added'] += 1
        self.stats['training_steps'] += 1
        
        return self.privacy_manager.apply_privacy(gradients)
    
    def get_stats(self) -> Dict:
        """Get training statistics."""
        return {
            **self.stats,
            'privacy_budget': self.privacy_manager.get_privacy_status(),
        }


# Utility functions for privacy analysis

def compute_epsilon_from_delta(
    delta: float,
    clip_norm: float,
    noise_scale: float,
) -> float:
    """
    Compute epsilon from other privacy parameters.
    
    Args:
        delta: Failure probability
        clip_norm: L2 clipping threshold
        noise_scale: Standard deviation of Gaussian noise
        
    Returns:
        Epsilon value achieving (ε, δ)-DP
    """
    if noise_scale <= 0:
        return np.inf
    
    # From (ε, δ)-DP theorem: ε = sqrt(2 * ln(1.25/δ)) * clip_norm / noise_scale
    return np.sqrt(2.0 * np.log(1.25 / delta)) * clip_norm / noise_scale


def compute_delta_from_epsilon(
    epsilon: float,
    clip_norm: float,
    noise_scale: float,
) -> float:
    """
    Compute delta from other privacy parameters.
    
    Args:
        epsilon: Privacy budget
        clip_norm: L2 clipping threshold
        noise_scale: Standard deviation of Gaussian noise
        
    Returns:
        Delta value achieving (ε, δ)-DP
    """
    if epsilon <= 0 or noise_scale <= 0:
        return 1.0
    
    # From (ε, δ)-DP theorem: δ = 1.25 * exp(-ε * noise_scale / clip_norm)²
    exponent = -epsilon * noise_scale / clip_norm
    return 1.25 * np.exp(exponent * exponent)


def estimate_noise_scale(
    epsilon: float,
    delta: float,
    clip_norm: float,
) -> float:
    """
    Estimate required noise scale for (ε, δ)-DP.
    
    Args:
        epsilon: Privacy budget
        delta: Failure probability
        clip_norm: L2 clipping threshold
        
    Returns:
        Recommended noise standard deviation
    """
    noise_multiplier = np.sqrt(2.0 * np.log(1.25 / delta))
    return noise_multiplier * clip_norm / (epsilon + 1e-10)


# Privacy budget accounting for different composition theorems

def advanced_composition_epsilon(
    epsilon_per_round: float,
    num_rounds: int,
    delta: float,
    delta_0: float = 1e-7,
) -> float:
    """
    Compute total epsilon under advanced composition theorem.
    
    Advanced composition gives tighter bounds than basic composition
    for adaptive mechanisms.
    
    Args:
        epsilon_per_round: Privacy budget per round
        num_rounds: Number of rounds
        delta: Total delta
        delta_0: Auxiliary delta parameter (typically small)
        
    Returns:
        Total epsilon after num_rounds
    """
    # ε_total = 2 * sqrt(ln(1/δ_0) * num_rounds) * ε_round + num_rounds * (exp(ε_round) - 1)
    term1 = 2 * np.sqrt(np.log(1.0 / delta_0) * num_rounds) * epsilon_per_round
    term2 = num_rounds * (np.exp(epsilon_per_round) - 1)
    return min(term1 + term2, delta)  # Cap at delta


def basic_composition_epsilon(
    epsilon_per_round: float,
    num_rounds: int,
) -> float:
    """
    Compute total epsilon under basic composition theorem.
    
    Basic composition: ε_total = num_rounds * ε_round
    
    Args:
        epsilon_per_round: Privacy budget per round
        num_rounds: Number of rounds
        
    Returns:
        Total epsilon after num_rounds
    """
    return num_rounds * epsilon_per_round
