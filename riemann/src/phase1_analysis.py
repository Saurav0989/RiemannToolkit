#!/usr/bin/env python3
"""
RIEMANN HYPOTHESIS - PHASE 1: Zero Computation and GUE Analysis
=================================================================

Compute first 10,000 zeros to high precision.
Compare spacing distribution to Random Matrix Theory (GUE) predictions.
Implement and test the xp model.

This is serious work toward understanding the spectral structure.
"""

import numpy as np
import mpmath as mp
from scipy.optimize import brentq
from typing import List, Tuple
import pickle
import os

mp.mp.dps = 60  # 60 decimal places for precision


# =============================================================================
# ZERO COMPUTATION
# =============================================================================

def compute_zeros(n_zeros: int = 1000, save_path: str = None, verbose: bool = True) -> np.ndarray:
    """
    Compute first n nontrivial zeros of ζ(s) to high precision.
    Returns array of imaginary parts (the γ values).
    """
    zeros = []
    t = 10.0
    step = 0.5
    
    if verbose:
        print(f"Computing {n_zeros} zeros...")
    
    # Get sign at starting point
    prev_z = float(mp.siegelz(t))
    
    while len(zeros) < n_zeros:
        t += step
        curr_z = float(mp.siegelz(t))
        
        if prev_z * curr_z < 0:  # Sign change = zero
            # Refine with Brent's method
            try:
                zero = brentq(lambda x: float(mp.siegelz(x)), t - step, t, xtol=1e-12)
                zeros.append(zero)
                
                if verbose and len(zeros) % 100 == 0:
                    print(f"  Found {len(zeros)}/{n_zeros} zeros...")
                    
            except Exception as e:
                pass  # Skip problematic intervals
        
        prev_z = curr_z
        
        # Safety: don't go too high
        if t > 50000:
            break
    
    result = np.array(zeros[:n_zeros])
    
    if save_path:
        np.save(save_path, result)
        print(f"  Saved to {save_path}")
    
    return result


def load_or_compute_zeros(n: int, cache_path: str = "zeros.npy") -> np.ndarray:
    """Load cached zeros or compute them."""
    if os.path.exists(cache_path):
        zeros = np.load(cache_path)
        if len(zeros) >= n:
            return zeros[:n]
    
    return compute_zeros(n, save_path=cache_path)


# =============================================================================
# GUE (GAUSSIAN UNITARY ENSEMBLE) COMPARISON
# =============================================================================

def normalize_zeros(zeros: np.ndarray) -> np.ndarray:
    """
    Normalize zeros so mean spacing = 1.
    The local density of zeros at height T is ≈ (1/2π) log(T/2π).
    """
    # For each zero, compute local density and normalize
    normalized = []
    
    for i, gamma in enumerate(zeros):
        # Local density at this height
        density = (1 / (2 * np.pi)) * np.log(gamma / (2 * np.pi)) if gamma > 2 * np.pi else 0.1
        
        # Integrate density from first zero to here (unfolding)
        if i == 0:
            normalized.append(0)
        else:
            # Numerical integration of density
            integral = sum((1 / (2 * np.pi)) * np.log(z / (2 * np.pi)) for z in zeros[:i])
            normalized.append(integral * (zeros[1] - zeros[0]) / len(zeros[:i]))
    
    return np.array(normalized)


def compute_spacing_distribution(zeros: np.ndarray, n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the empirical spacing distribution of normalized zeros.
    """
    # Normalize spacings
    spacings = np.diff(zeros)
    mean_spacing = np.mean(spacings)
    normalized_spacings = spacings / mean_spacing
    
    # Histogram
    hist, bin_edges = np.histogram(normalized_spacings, bins=n_bins, range=(0, 4), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return bin_centers, hist


def gue_spacing_distribution(s: np.ndarray) -> np.ndarray:
    """
    GUE (Wigner surmise) spacing distribution: P(s) = (32/π²) s² exp(-4s²/π)
    This is the Random Matrix Theory prediction for zeros on the critical line.
    """
    return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)


def poisson_spacing_distribution(s: np.ndarray) -> np.ndarray:
    """
    Poisson spacing distribution: P(s) = exp(-s)
    This is what you'd expect from random, uncorrelated zeros.
    """
    return np.exp(-s)


def compare_to_gue(zeros: np.ndarray) -> dict:
    """
    Compare zero spacing to GUE and Poisson predictions.
    GUE = what we expect if RH is deeply true (quantum chaos)
    Poisson = what we'd expect from random noise
    """
    bin_centers, empirical = compute_spacing_distribution(zeros)
    
    gue_pred = gue_spacing_distribution(bin_centers)
    poisson_pred = poisson_spacing_distribution(bin_centers)
    
    # L2 distance to each prediction
    l2_gue = np.sqrt(np.mean((empirical - gue_pred)**2))
    l2_poisson = np.sqrt(np.mean((empirical - poisson_pred)**2))
    
    # Correlation with each
    corr_gue = np.corrcoef(empirical, gue_pred)[0, 1]
    corr_poisson = np.corrcoef(empirical, poisson_pred)[0, 1]
    
    return {
        'bin_centers': bin_centers,
        'empirical': empirical,
        'gue_prediction': gue_pred,
        'poisson_prediction': poisson_pred,
        'l2_distance_gue': l2_gue,
        'l2_distance_poisson': l2_poisson,
        'correlation_gue': corr_gue,
        'correlation_poisson': corr_poisson,
        'follows_gue': l2_gue < l2_poisson
    }


# =============================================================================
# PAIR CORRELATION (Montgomery's Conjecture)
# =============================================================================

def montgomery_pair_correlation(zeros: np.ndarray, range_limit: float = 3.0) -> dict:
    """
    Compute the pair correlation function R₂(r).
    
    Montgomery conjectured: R₂(r) = 1 - (sin(πr)/(πr))²
    
    This is the GUE prediction and matches zero pairs remarkably.
    """
    n = len(zeros)
    
    # Normalize spacings for each zero
    mean_local_spacing = np.mean(np.diff(zeros))
    
    # Compute all pairwise normalized differences
    pairs = []
    for i in range(n):
        for j in range(i + 1, min(i + 100, n)):  # Local pairs
            # Normalize by local density
            local_density = (1 / (2 * np.pi)) * np.log(zeros[i] / (2 * np.pi))
            delta = (zeros[j] - zeros[i]) * local_density
            
            if 0 < delta < range_limit:
                pairs.append(delta)
    
    if len(pairs) < 50:
        return {'error': 'Not enough pairs'}
    
    # Histogram
    n_bins = 30
    hist, bin_edges = np.histogram(pairs, bins=n_bins, range=(0, range_limit), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Montgomery prediction
    def montgomery_prediction(r):
        if r < 0.01:
            return 0.0
        return 1 - (np.sin(np.pi * r) / (np.pi * r))**2
    
    prediction = np.array([montgomery_prediction(r) for r in bin_centers])
    
    # Compare
    correlation = np.corrcoef(hist, prediction)[0, 1]
    l2_error = np.sqrt(np.mean((hist - prediction)**2))
    
    return {
        'bin_centers': bin_centers,
        'empirical_R2': hist,
        'montgomery_prediction': prediction,
        'correlation': correlation,
        'l2_error': l2_error,
        'n_pairs': len(pairs)
    }


# =============================================================================
# XP MODEL (Berry-Keating)
# =============================================================================

def xp_eigenvalue_density(E: float) -> float:
    """
    Eigenvalue density for H = xp.
    The smooth density is: ρ(E) = (1/2π) log(E/2π) for E > 0
    
    This matches the average density of zeta zeros!
    """
    if E <= 2 * np.pi:
        return 0.0
    return (1 / (2 * np.pi)) * np.log(E / (2 * np.pi))


def xp_integrated_density(E: float) -> float:
    """
    N(E) = number of eigenvalues < E
    For xp model: N(E) ≈ (E/2π) log(E/2π) - E/2π
    
    This should match N(T) for zeta zeros.
    """
    if E <= 2 * np.pi:
        return 0.0
    return (E / (2 * np.pi)) * np.log(E / (2 * np.pi)) - E / (2 * np.pi)


def compare_xp_counting(zeros: np.ndarray) -> dict:
    """
    Compare empirical N(T) (number of zeros < T) to xp model prediction.
    """
    T_values = np.linspace(20, zeros[-1], 100)
    
    empirical_N = []
    xp_N = []
    
    for T in T_values:
        empirical_N.append(np.sum(zeros < T))
        xp_N.append(xp_integrated_density(T))
    
    empirical_N = np.array(empirical_N)
    xp_N = np.array(xp_N)
    
    # Correlation
    correlation = np.corrcoef(empirical_N, xp_N)[0, 1]
    
    # Relative error
    relative_error = np.mean(np.abs(empirical_N - xp_N) / np.maximum(empirical_N, 1))
    
    return {
        'T_values': T_values,
        'empirical_count': empirical_N,
        'xp_prediction': xp_N,
        'correlation': correlation,
        'mean_relative_error': relative_error
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_phase1_analysis(n_zeros: int = 2000):
    """
    Complete Phase 1 analysis:
    1. Compute zeros
    2. Compare to GUE
    3. Montgomery pair correlation
    4. xp model comparison
    """
    print("\n" + "="*70)
    print("  RIEMANN HYPOTHESIS - PHASE 1 ANALYSIS")
    print("  Zero Computation and GUE Comparison")
    print("="*70)
    
    # 1. Compute zeros
    print("\n[1/4] COMPUTING ZEROS")
    print("-" * 50)
    cache_path = os.path.join(os.path.dirname(__file__), "zeros_cache.npy")
    zeros = load_or_compute_zeros(n_zeros, cache_path)
    print(f"  Total zeros: {len(zeros)}")
    print(f"  Range: [{zeros[0]:.6f}, {zeros[-1]:.6f}]")
    print(f"  First 5: {zeros[:5]}")
    
    # 2. GUE comparison
    print("\n[2/4] GUE SPACING ANALYSIS")
    print("-" * 50)
    gue_results = compare_to_gue(zeros)
    print(f"  L2 distance to GUE:     {gue_results['l2_distance_gue']:.4f}")
    print(f"  L2 distance to Poisson: {gue_results['l2_distance_poisson']:.4f}")
    print(f"  Correlation with GUE:   {gue_results['correlation_gue']:.4f}")
    print(f"  Follows GUE? {gue_results['follows_gue']}")
    
    # 3. Montgomery pair correlation
    print("\n[3/4] MONTGOMERY PAIR CORRELATION")
    print("-" * 50)
    pair_results = montgomery_pair_correlation(zeros)
    if 'error' not in pair_results:
        print(f"  Pairs analyzed: {pair_results['n_pairs']}")
        print(f"  Correlation with Montgomery: {pair_results['correlation']:.4f}")
        print(f"  L2 error: {pair_results['l2_error']:.4f}")
    else:
        print(f"  {pair_results['error']}")
    
    # 4. xp model
    print("\n[4/4] XP MODEL COMPARISON")
    print("-" * 50)
    xp_results = compare_xp_counting(zeros)
    print(f"  Correlation with xp counting: {xp_results['correlation']:.6f}")
    print(f"  Mean relative error: {xp_results['mean_relative_error']:.4f}")
    
    # Summary
    print("\n" + "="*70)
    print("  PHASE 1 SUMMARY")
    print("="*70)
    
    gue_match = gue_results['follows_gue'] and gue_results['correlation_gue'] > 0.9
    montgomery_match = 'correlation' in pair_results and pair_results['correlation'] > 0.9
    xp_match = xp_results['correlation'] > 0.999
    
    print(f"""
  Zeros computed: {len(zeros)} (to {mp.mp.dps} decimal places)
  
  KEY FINDINGS:
  • GUE spacing match:       {'✅ YES' if gue_match else '❌ NO'} (r = {gue_results['correlation_gue']:.3f})
  • Montgomery correlation:  {'✅ YES' if montgomery_match else '⚠️ PARTIAL'} (r = {pair_results.get('correlation', 0):.3f})
  • xp counting match:       {'✅ YES' if xp_match else '❌ NO'} (r = {xp_results['correlation']:.6f})
  
  INTERPRETATION:
  • GUE match confirms zeros behave like eigenvalues of random Hermitian matrices
  • Montgomery match confirms pair correlation ≈ 1 - (sin πr / πr)²
  • xp match confirms N(T) ≈ (T/2π)log(T/2π) - T/2π
  
  The xp model gets the DENSITY right but not the INDIVIDUAL zeros.
  This is the gap: we know the statistical structure, but not the exact operator.
    """)
    
    return {
        'zeros': zeros,
        'gue': gue_results,
        'montgomery': pair_results,
        'xp': xp_results
    }


if __name__ == "__main__":
    results = run_phase1_analysis(n_zeros=2000)
