#!/usr/bin/env python3
"""
MERTENS FUNCTION: RH Equivalent Formulation
=============================================

RH Attack Plan - Vector 3

RH is equivalent to: |M(n)| < n^(1/2 + ε) for all ε > 0, sufficiently large n

where M(n) = Σ_{k=1}^{n} μ(k), μ = Möbius function

If we can show M(n) = O(n^(1/2 + ε)) for all ε > 0, we prove RH.

This implementation:
1. Computes M(n) efficiently via sieve
2. Tests the bound |M(n)| / n^0.5
3. Looks for patterns in M(n) growth
"""

import numpy as np
from typing import List, Tuple
import time


# =============================================================================
# MÖBIUS FUNCTION SIEVE
# =============================================================================

def mobius_sieve(n_max: int) -> np.ndarray:
    """
    Compute μ(k) for k = 0, 1, ..., n_max using sieve.
    
    μ(n) = (-1)^k if n is product of k distinct primes
         = 0     if n has a squared prime factor
    """
    # Initialize
    mu = np.ones(n_max + 1, dtype=np.int8)
    mu[0] = 0
    
    # Factor counts for detecting square factors
    smallest_prime = np.zeros(n_max + 1, dtype=np.int32)
    
    # Sieve
    for p in range(2, n_max + 1):
        if smallest_prime[p] == 0:  # p is prime
            # Mark multiples
            for m in range(p, n_max + 1, p):
                if smallest_prime[m] == 0:
                    smallest_prime[m] = p
                
                # Check if p^2 divides m
                if (m // p) % p == 0:
                    mu[m] = 0
                else:
                    mu[m] *= -1
    
    return mu


def mertens_function(n_max: int) -> np.ndarray:
    """
    Compute M(n) = Σ_{k=1}^{n} μ(k) for n = 0, 1, ..., n_max
    """
    mu = mobius_sieve(n_max)
    M = np.cumsum(mu)
    return M


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_mertens(n_max: int = 10**6) -> dict:
    """
    Analyze Mertens function and its relation to RH.
    """
    print(f"Computing M(n) for n up to {n_max:,}...")
    start = time.time()
    
    M = mertens_function(n_max)
    
    elapsed = time.time() - start
    print(f"  Computed in {elapsed:.1f}s")
    
    # Compute |M(n)| / n^0.5 ratio
    n_vals = np.arange(1, n_max + 1)
    ratios = np.abs(M[1:]) / np.sqrt(n_vals)
    
    # Find maximum ratio (would need to be unbounded if RH is false)
    max_ratio = np.max(ratios)
    max_idx = np.argmax(ratios) + 1
    
    # Sample values at powers of 10
    samples = []
    for exp in range(1, int(np.log10(n_max)) + 1):
        n = 10**exp
        if n <= n_max:
            samples.append({
                'n': n,
                'M(n)': M[n],
                '|M(n)|/√n': abs(M[n]) / np.sqrt(n)
            })
    
    return {
        'n_max': n_max,
        'max_ratio': max_ratio,
        'max_at': max_idx,
        'samples': samples,
        'M_values': M
    }


def test_rh_bound(M: np.ndarray, epsilon: float = 0.01) -> dict:
    """
    Test if |M(n)| < n^(0.5 + ε) for all n.
    
    If this holds for all ε > 0, RH is true.
    """
    n_max = len(M) - 1
    n_vals = np.arange(1, n_max + 1)
    
    # Bound: n^(0.5 + ε)
    bounds = n_vals ** (0.5 + epsilon)
    
    # Check violations
    violations = np.where(np.abs(M[1:]) > bounds)[0] + 1
    
    return {
        'epsilon': epsilon,
        'n_max': n_max,
        'violations': len(violations),
        'first_violation': violations[0] if len(violations) > 0 else None,
        'bound_holds': len(violations) == 0
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*60)
    print("  MERTENS FUNCTION ANALYSIS")
    print("  RH ⟺ |M(n)| < n^(0.5 + ε) for all ε > 0")
    print("="*60)
    
    # Analyze for n up to 10^6
    results = analyze_mertens(n_max=10**6)
    
    print("\n[1/2] MERTENS VALUES AT POWERS OF 10")
    print("-" * 50)
    for s in results['samples']:
        print(f"  n = {s['n']:>10,}: M(n) = {s['M(n)']:>8,}, |M|/√n = {s['|M(n)|/√n']:.4f}")
    
    print(f"\n  Maximum |M(n)|/√n = {results['max_ratio']:.4f} at n = {results['max_at']:,}")
    
    # Test RH bounds
    print("\n[2/2] TESTING RH BOUNDS")
    print("-" * 50)
    
    M = results['M_values']
    
    for eps in [0.1, 0.05, 0.01, 0.001]:
        bound_result = test_rh_bound(M, epsilon=eps)
        status = "✅ HOLDS" if bound_result['bound_holds'] else f"❌ FAILS at n={bound_result['first_violation']}"
        print(f"  ε = {eps:.3f}: |M(n)| < n^({0.5+eps:.3f}) → {status}")
    
    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"""
  M(n) computed to n = {results['n_max']:,}
  
  Maximum ratio |M(n)|/√n = {results['max_ratio']:.4f}
  
  Observation: The ratio stays bounded (doesn't grow without limit)
  This is CONSISTENT with RH but doesn't prove it.
  
  To falsify RH, we'd need |M(n)|/√n → ∞ as n → ∞
  Current data: No evidence of that.
    """)
    
    return results


if __name__ == "__main__":
    results = main()
