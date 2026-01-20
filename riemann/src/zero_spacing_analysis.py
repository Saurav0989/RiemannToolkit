#!/usr/bin/env python3
"""
ZERO SPACING ANALYSIS: Deep Statistical Study
===============================================

RH Attack Plan - Vector 4: Random Matrix Theory

Montgomery's Pair Correlation Conjecture:
The pair correlation of zeta zeros matches GUE eigenvalue statistics.

This implements:
1. Zero spacing distribution
2. GUE (Wigner surmise) comparison
3. Pair correlation function R₂(r)
4. Spectral rigidity Δ₃(L)
"""

import numpy as np
from typing import List, Tuple, Dict
import sys
sys.path.insert(0, 'riemann/src')


# =============================================================================
# SPACING DISTRIBUTION
# =============================================================================

def compute_normalized_spacings(zeros: np.ndarray) -> np.ndarray:
    """
    Compute normalized spacings between consecutive zeros.
    
    The local mean spacing at height T is ~2π/log(T/2π).
    We normalize so mean spacing = 1.
    """
    spacings = np.diff(zeros)
    
    # Local normalization using density
    normalized = []
    for i, (z, s) in enumerate(zip(zeros[:-1], spacings)):
        # Local density: ρ(t) ≈ (1/2π) log(t/2π)
        if z > 2 * np.pi:
            local_density = (1 / (2 * np.pi)) * np.log(z / (2 * np.pi))
            normalized.append(s * local_density)
        else:
            normalized.append(s)
    
    return np.array(normalized)


def gue_wigner_surmise(s: np.ndarray) -> np.ndarray:
    """
    GUE (Gaussian Unitary Ensemble) spacing distribution.
    
    P(s) = (32/π²) s² exp(-4s²/π)
    
    This is what we expect if zeros behave like random matrix eigenvalues.
    """
    return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)


def poisson_distribution(s: np.ndarray) -> np.ndarray:
    """
    Poisson spacing distribution: P(s) = exp(-s)
    
    This is what we'd expect from independent random points.
    """
    return np.exp(-s)


def analyze_spacing_distribution(zeros: np.ndarray) -> Dict:
    """
    Full spacing distribution analysis.
    """
    normalized = compute_normalized_spacings(zeros)
    
    # Histogram
    bins = np.linspace(0, 4, 51)
    hist, bin_edges = np.histogram(normalized, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Predictions
    gue_pred = gue_wigner_surmise(bin_centers)
    poisson_pred = poisson_distribution(bin_centers)
    
    # Fit metrics
    l2_gue = np.sqrt(np.mean((hist - gue_pred)**2))
    l2_poisson = np.sqrt(np.mean((hist - poisson_pred)**2))
    
    corr_gue = np.corrcoef(hist, gue_pred)[0, 1]
    corr_poisson = np.corrcoef(hist, poisson_pred)[0, 1]
    
    return {
        'normalized_spacings': normalized,
        'histogram': hist,
        'bin_centers': bin_centers,
        'gue_prediction': gue_pred,
        'poisson_prediction': poisson_pred,
        'l2_gue': l2_gue,
        'l2_poisson': l2_poisson,
        'corr_gue': corr_gue,
        'corr_poisson': corr_poisson,
        'follows_gue': l2_gue < l2_poisson
    }


# =============================================================================
# PAIR CORRELATION
# =============================================================================

def pair_correlation_function(zeros: np.ndarray, r_max: float = 3.0, n_bins: int = 30) -> Dict:
    """
    Compute the pair correlation function R₂(r).
    
    Montgomery's conjecture: R₂(r) = 1 - (sin(πr)/(πr))²
    """
    n = len(zeros)
    
    # Collect normalized pair differences
    pairs = []
    
    for i in range(min(n, 2000)):  # Limit for performance
        # Local density at zeros[i]
        t = zeros[i]
        if t < 10:
            continue
        local_density = (1 / (2 * np.pi)) * np.log(t / (2 * np.pi))
        
        # Compare to nearby zeros
        for j in range(i + 1, min(i + 100, n)):
            delta = (zeros[j] - zeros[i]) * local_density
            if delta < r_max:
                pairs.append(delta)
    
    pairs = np.array(pairs)
    
    # Histogram
    hist, bin_edges = np.histogram(pairs, bins=n_bins, range=(0, r_max), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Montgomery prediction
    def R2_prediction(r):
        if r < 0.01:
            return 0.0
        return 1 - (np.sin(np.pi * r) / (np.pi * r))**2
    
    montgomery = np.array([R2_prediction(r) for r in bin_centers])
    
    # Correlation
    corr = np.corrcoef(hist, montgomery)[0, 1]
    
    return {
        'bin_centers': bin_centers,
        'empirical_R2': hist,
        'montgomery_prediction': montgomery,
        'correlation': corr,
        'n_pairs': len(pairs)
    }


# =============================================================================
# SPECTRAL RIGIDITY
# =============================================================================

def spectral_rigidity_delta3(zeros: np.ndarray, L_values: List[float] = None) -> Dict:
    """
    Compute the spectral rigidity Δ₃(L).
    
    Δ₃(L) measures how well the number of zeros in an interval of length L
    follows a "staircase" pattern.
    
    For GUE: Δ₃(L) → (1/π²) log(L) for large L
    For Poisson: Δ₃(L) = L/15
    """
    if L_values is None:
        L_values = [1, 2, 5, 10, 20, 50]
    
    n = len(zeros)
    
    # Normalize zeros (unfold)
    normalized = compute_normalized_spacings(zeros)
    unfolded = np.cumsum(np.concatenate([[0], normalized]))
    
    results = {}
    
    for L in L_values:
        # Compute Δ₃ for this L
        # Sample many intervals
        deltas = []
        
        for _ in range(min(1000, n // 10)):
            # Random starting point
            start_idx = np.random.randint(0, max(1, n - int(L * 2)))
            
            # Count zeros in interval [a, a+L] for unfolded sequence
            a = unfolded[start_idx]
            b = a + L
            
            # Linear fit to staircase
            in_range = (unfolded >= a) & (unfolded <= b)
            if np.sum(in_range) < 3:
                continue
            
            x = unfolded[in_range] - a
            y = np.arange(np.sum(in_range))
            
            # Fit line y = mx + c
            if len(x) > 1:
                m, c = np.polyfit(x, y, 1)
                residuals = y - (m * x + c)
                delta = np.var(residuals)
                deltas.append(delta)
        
        if deltas:
            results[L] = np.mean(deltas)
    
    # GUE prediction: Δ₃(L) ≈ (1/π²) log(L)
    gue_pred = {L: (1/np.pi**2) * np.log(L) for L in L_values if L > 0}
    
    # Poisson prediction: Δ₃(L) = L/15
    poisson_pred = {L: L/15 for L in L_values}
    
    return {
        'L_values': L_values,
        'delta3': results,
        'gue_prediction': gue_pred,
        'poisson_prediction': poisson_pred
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("  ZERO SPACING ANALYSIS")
    print("  Comparing to GUE / Random Matrix Theory")
    print("="*70)
    
    # Load or compute zeros
    from riemann_siegel import find_zeros_in_range
    
    print("\n[1/4] COMPUTING ZEROS")
    print("-" * 50)
    
    # Use cached zeros if available
    import os
    cache_path = 'riemann/src/zeros_10k.npy'
    
    if os.path.exists(cache_path):
        zeros = np.load(cache_path)
        print(f"  Loaded {len(zeros)} zeros from cache")
    else:
        print("  Computing zeros in [14, 10000]...")
        zeros = np.array(find_zeros_in_range(14, 10000, step=0.3))
        np.save(cache_path, zeros)
        print(f"  Found {len(zeros)} zeros, saved to cache")
    
    # Spacing distribution
    print("\n[2/4] SPACING DISTRIBUTION")
    print("-" * 50)
    
    spacing_results = analyze_spacing_distribution(zeros)
    
    print(f"  L2 distance to GUE:     {spacing_results['l2_gue']:.4f}")
    print(f"  L2 distance to Poisson: {spacing_results['l2_poisson']:.4f}")
    print(f"  Correlation with GUE:   {spacing_results['corr_gue']:.4f}")
    print(f"  Follows GUE? {spacing_results['follows_gue']}")
    
    # Pair correlation
    print("\n[3/4] PAIR CORRELATION (Montgomery)")
    print("-" * 50)
    
    pair_results = pair_correlation_function(zeros)
    
    print(f"  Pairs analyzed: {pair_results['n_pairs']}")
    print(f"  Correlation with Montgomery: {pair_results['correlation']:.4f}")
    
    # Spectral rigidity
    print("\n[4/4] SPECTRAL RIGIDITY Δ₃(L)")
    print("-" * 50)
    
    rigidity = spectral_rigidity_delta3(zeros)
    
    print("  L     Δ₃(L)    GUE pred   Poisson pred")
    for L in rigidity['L_values']:
        if L in rigidity['delta3']:
            d3 = rigidity['delta3'][L]
            gue = rigidity['gue_prediction'].get(L, 0)
            poi = rigidity['poisson_prediction'].get(L, 0)
            print(f"  {L:4.0f}  {d3:7.4f}   {gue:7.4f}    {poi:7.4f}")
    
    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    
    gue_match = spacing_results['corr_gue'] > 0.95
    montgomery_match = pair_results['correlation'] > 0.90
    
    print(f"""
  RESULTS:
  • Spacing follows GUE:      {'✅ YES' if gue_match else '❌ NO'} (r = {spacing_results['corr_gue']:.3f})
  • Montgomery correlation:   {'✅ YES' if montgomery_match else '❌ NO'} (r = {pair_results['correlation']:.3f})
  
  INTERPRETATION:
  • Zeros behave like eigenvalues of random Hermitian matrices (GUE)
  • This is strong evidence FOR the Riemann Hypothesis
  • The spectral structure supports Hilbert-Pólya conjecture
  
  Note: Statistical evidence ≠ Proof. But this is as expected if RH is true.
    """)
    
    return {
        'zeros': zeros,
        'spacing': spacing_results,
        'pair_correlation': pair_results,
        'rigidity': rigidity
    }


if __name__ == "__main__":
    results = main()
