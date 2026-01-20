#!/usr/bin/env python3
"""
OFF-CRITICAL-LINE BOUND EXPERIMENT
===================================

From Brother's Proof Strategy:
"If you can find a bound B(σ) such that |ζ(σ + it)| > B(σ) > 0 for σ ≠ 0.5,
and PROVE that bound exists, you're close to RH."

This experiment:
1. For each σ ≠ 0.5, compute minimum |ζ(σ + it)| over many t
2. Look for a pattern in these minimums
3. If minimums stay bounded away from 0, this suggests a theorem
"""

import numpy as np
import mpmath
mpmath.mp.dps = 30

from typing import Dict, List, Tuple


def compute_zeta_minimum(sigma: float, t_range: Tuple[float, float], 
                         n_samples: int = 10000) -> Dict:
    """
    For Re(s) = σ, find the minimum |ζ(σ + it)| over t in t_range.
    
    If RH is true and σ ≠ 0.5, this minimum should be bounded away from 0.
    """
    t_vals = np.linspace(t_range[0], t_range[1], n_samples)
    
    min_value = float('inf')
    min_t = 0
    
    for t in t_vals:
        s = mpmath.mpc(sigma, t)
        zeta_val = abs(mpmath.zeta(s))
        
        if zeta_val < min_value:
            min_value = zeta_val
            min_t = t
    
    return {
        'sigma': sigma,
        't_range': t_range,
        'min_zeta': float(min_value),
        'min_t': float(min_t)
    }


def search_for_bound_B(sigma_values: List[float], t_max: float = 1000) -> Dict:
    """
    For each σ, find minimum |ζ(σ + it)| and look for pattern B(σ).
    """
    results = {}
    
    print(f"Searching for bound B(σ) over t ∈ [10, {t_max}]")
    print("-" * 50)
    
    for sigma in sigma_values:
        result = compute_zeta_minimum(sigma, (10, t_max), n_samples=2000)
        results[sigma] = result
        
        print(f"  σ = {sigma:.2f}: min|ζ| = {result['min_zeta']:.6f} at t = {result['min_t']:.2f}")
    
    return results


def analyze_bound_pattern(results: Dict) -> Dict:
    """
    Analyze if there's a pattern B(σ) bounding |ζ(σ + it)| away from 0.
    """
    sigmas = sorted(results.keys())
    min_vals = [results[s]['min_zeta'] for s in sigmas]
    
    # Check: does minimum approach 0 only near σ = 0.5?
    half_idx = min(range(len(sigmas)), key=lambda i: abs(sigmas[i] - 0.5))
    
    # Fit a polynomial in (σ - 0.5)²
    x = np.array([(s - 0.5)**2 for s in sigmas])
    y = np.array(min_vals)
    
    # Try B(σ) ≈ c * |σ - 0.5|^α
    x_log = np.log(np.abs(np.array(sigmas) - 0.5) + 1e-10)
    y_log = np.log(np.array(min_vals) + 1e-10)
    
    # Linear fit in log-log space
    valid = np.abs(np.array(sigmas) - 0.5) > 0.01
    if np.sum(valid) > 2:
        try:
            coef = np.polyfit(x_log[valid], y_log[valid], 1)
            alpha = coef[0]
            c = np.exp(coef[1])
            
            print(f"\n  Potential bound: B(σ) ≈ {c:.4f} * |σ - 0.5|^{alpha:.2f}")
        except:
            alpha, c = None, None
    else:
        alpha, c = None, None
    
    return {
        'pattern_found': alpha is not None,
        'exponent_alpha': alpha,
        'coefficient_c': c,
        'data': results
    }


def main():
    print("\n" + "="*60)
    print("  OFF-CRITICAL-LINE BOUND EXPERIMENT")
    print("  Testing: Does |ζ(σ + it)| stay bounded away from 0 for σ ≠ 0.5?")
    print("="*60)
    
    # Test σ values on both sides of critical line
    sigma_values = [0.1, 0.2, 0.3, 0.4, 0.45, 0.48, 0.52, 0.55, 0.6, 0.7, 0.8, 0.9]
    
    print("\n[1/3] COMPUTING MINIMUM |ζ| FOR EACH σ")
    results = search_for_bound_B(sigma_values, t_max=500)
    
    print("\n[2/3] ANALYZING PATTERN")
    print("-" * 50)
    
    analysis = analyze_bound_pattern(results)
    
    if analysis['pattern_found']:
        print(f"  Found potential bound pattern!")
        print(f"  B(σ) ≈ {analysis['coefficient_c']:.4f} * |σ - 0.5|^{analysis['exponent_alpha']:.2f}")
    else:
        print("  No clear pattern found")
    
    print("\n[3/3] INTERPRETATION")
    print("-" * 50)
    
    # Key observation
    min_at_half = min(results.values(), key=lambda r: abs(r['sigma'] - 0.5))
    
    print(f"""
  KEY OBSERVATIONS:
  
  1. At σ = 0.5 (critical line):
     min|ζ| = {min_at_half['min_zeta']:.6f}
     This should approach 0 (zeros are here!)
     
  2. Away from σ = 0.5:
     min|ζ| stays positive
     
  3. If we could PROVE |ζ(σ + it)| > B(σ) > 0 for σ ≠ 0.5:
     This would imply RH (no zeros off the line)
     
  NEXT STEP: Find a mathematical argument for why B(σ) > 0
    """)
    
    return analysis


if __name__ == "__main__":
    analysis = main()
