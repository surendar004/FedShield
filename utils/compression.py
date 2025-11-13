"""
Compression utilities for FedShield

Provides simple quantization and top-k sparsification helpers.
"""

from typing import Dict, List, Tuple
import numpy as np


def quantize_array(arr: np.ndarray, bits: int = 8) -> Tuple[np.ndarray, float, float]:
    """Quantize a numpy array to a fixed number of bits.

    Returns tuple: (quantized_array, min_val, scale)
    where quantized_array is dtype=np.uint8 (or larger depending on bits).
    """
    if bits <= 0 or bits > 16:
        raise ValueError("bits must be between 1 and 16")

    arr = np.asarray(arr)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if max_val == min_val:
        # constant array
        scale = 1.0
        q = np.zeros_like(arr, dtype=np.uint16 if bits > 8 else np.uint8)
        return q, min_val, scale

    levels = 2 ** bits - 1
    scale = (max_val - min_val) / levels
    q = np.round((arr - min_val) / scale).astype(np.uint16 if bits > 8 else np.uint8)
    return q, min_val, scale


def dequantize_array(q: np.ndarray, min_val: float, scale: float) -> np.ndarray:
    """Dequantize array produced by quantize_array."""
    q = np.asarray(q)
    return q.astype(np.float32) * scale + min_val


def top_k_sparsify(arr: np.ndarray, k_fraction: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Keep top-k_fraction absolute values, zero out others.

    Returns: (sparse_array, mask) where mask is boolean same-shape array.
    """
    arr = np.asarray(arr)
    if k_fraction <= 0 or k_fraction > 1:
        raise ValueError("k_fraction must be in (0,1]")
    flat = np.abs(arr).ravel()
    k = max(1, int(len(flat) * k_fraction))
    if k >= len(flat):
        mask = np.ones_like(arr, dtype=bool)
        return arr.copy(), mask
    thresh = np.partition(flat, -k)[-k]
    mask = np.abs(arr) >= thresh
    sparse = arr * mask
    return sparse, mask


def compress_weights(weights: Dict[str, List[np.ndarray]], bits: int = 8, k_fraction: float = 0.1) -> Dict:
    """Compress a model weight dict.

    Returns a dict with compressed payload and metadata for reconstruction.
    """
    payload = {}
    for key, lst in weights.items():
        if key in ['coefs', 'intercepts']:
            q_list = []
            meta_list = []
            for arr in lst:
                sparse, mask = top_k_sparsify(arr, k_fraction)
                q, min_v, scale = quantize_array(sparse, bits)
                q_list.append(q.tolist())
                meta_list.append({'min': min_v, 'scale': scale, 'mask': mask.tolist()})
            payload[key] = {'quantized': q_list, 'meta': meta_list}
        else:
            # other arrays (scaler_mean/scale) - quantize whole
            arr = np.asarray(lst)
            q, min_v, scale = quantize_array(arr, bits)
            payload[key] = {'quantized': q.tolist(), 'meta': {'min': min_v, 'scale': scale}}
    return payload


def decompress_weights(payload: Dict) -> Dict[str, List[np.ndarray]]:
    """Decompress payload created by compress_weights."""
    weights = {}
    for key, val in payload.items():
        if key in ['coefs', 'intercepts']:
            out_list = []
            for q_list, meta in zip(val['quantized'], val['meta']):
                q = np.asarray(q_list)
                min_v = meta['min']
                scale = meta['scale']
                mask = np.asarray(meta['mask'], dtype=bool)
                arr = dequantize_array(q, min_v, scale)
                # apply mask
                arr = arr * mask
                out_list.append(arr)
            weights[key] = out_list
        else:
            q = np.asarray(val['quantized'])
            min_v = val['meta']['min']
            scale = val['meta']['scale']
            arr = dequantize_array(q, min_v, scale)
            weights[key] = arr
    return weights
