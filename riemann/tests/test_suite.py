#!/usr/bin/env python3
"""
RIEMANN TOOLKIT: Comprehensive Test Suite
===========================================

Run all verification tests and report overall RH consistency.
"""

import numpy as np
import sys
import time

sys.path.insert(0, 'riemann/src')


def run_all_tests():
    print("\n" + "🔬"*30)
    print("  RIEMANN TOOLKIT: COMPREHENSIVE TEST SUITE")
    print("🔬"*30)
    
    results = {}
    
    # Test 1: Riemann-Siegel Zero Finder
    print("\n[1/5] RIEMANN-SIEGEL ZERO FINDER")
    print("-" * 50)
    
    from riemann_siegel import find_zeros_in_range, validate_implementation
    
    val = validate_implementation()
    print(f"  Validation: {val['all_found']}")
    print(f"  Max error: {val['max_error']:.2e}")
    
    results['rs_valid'] = val['all_found'] and val['max_error'] < 1
    
    # Test 2: Mertens Function
    print("\n[2/5] MERTENS FUNCTION BOUNDS")
    print("-" * 50)
    
    from mertens import mertens_function, test_rh_bound
    
    M = mertens_function(100000)
    bound_result = test_rh_bound(M, epsilon=0.01)
    print(f"  ε = 0.01: {bound_result['bound_holds']}")
    
    results['mertens_valid'] = bound_result['bound_holds']
    
    # Test 3: GUE Correlation
    print("\n[3/5] GUE SPACING STATISTICS")
    print("-" * 50)
    
    # Load cached zeros
    zeros = np.load('riemann/src/zeros_cache.npy')[:1000]
    
    spacings = np.diff(zeros)
    normalized = spacings / np.mean(spacings)
    
    def gue(s):
        return (32/np.pi**2) * s**2 * np.exp(-4*s**2/np.pi)
    
    bins = np.linspace(0, 4, 41)
    hist, edges = np.histogram(normalized, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    
    corr = np.corrcoef(hist, gue(centers))[0, 1]
    print(f"  GUE correlation: r = {corr:.4f}")
    
    results['gue_valid'] = corr > 0.95
    
    # Test 4: Montgomery Pair Correlation
    print("\n[4/5] MONTGOMERY PAIR CORRELATION")
    print("-" * 50)
    
    mean_spacing = np.mean(spacings)
    pairs = []
    
    for i in range(min(500, len(zeros))):
        for j in range(i+1, min(i+30, len(zeros))):
            d = (zeros[j] - zeros[i]) / mean_spacing
            if d < 3:
                pairs.append(d)
    
    hist2, edges2 = np.histogram(pairs, bins=30, range=(0,3), density=True)
    centers2 = (edges2[:-1] + edges2[1:]) / 2
    
    def R2(r):
        if r < 0.01: return 0
        return 1 - (np.sin(np.pi*r)/(np.pi*r))**2
    
    mont_corr = np.corrcoef(hist2, np.array([R2(r) for r in centers2]))[0, 1]
    print(f"  Montgomery: r = {mont_corr:.4f}")
    
    results['montgomery_valid'] = mont_corr > 0.9
    
    # Test 5: Prime Counting (for x >= 2657)
    print("\n[5/5] PRIME COUNTING BOUND (x ≥ 2657)")
    print("-" * 50)
    
    from prime_counting import prime_counting, logarithmic_integral, rh_error_bound
    
    pi = prime_counting(100000)
    
    # Test only for x >= 2657
    test_x = [3000, 5000, 10000, 50000, 100000]
    violations = 0
    
    for x in test_x:
        error = abs(pi[x] - logarithmic_integral(x))
        bound = rh_error_bound(x)
        if error > bound:
            violations += 1
    
    print(f"  Violations (x ≥ 2657): {violations}/{len(test_x)}")
    
    results['prime_counting_valid'] = violations == 0
    
    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    
    all_pass = all(results.values())
    
    print("\n  | Test                  | Status |")
    print("  |-----------------------|--------|")
    print(f"  | Riemann-Siegel        | {'✅' if results['rs_valid'] else '❌'}      |")
    print(f"  | Mertens Bound         | {'✅' if results['mertens_valid'] else '❌'}      |")
    print(f"  | GUE Spacing           | {'✅' if results['gue_valid'] else '❌'}      |")
    print(f"  | Montgomery Pair       | {'✅' if results['montgomery_valid'] else '❌'}      |")
    print(f"  | Prime Counting        | {'✅' if results['prime_counting_valid'] else '❌'}      |")
    
    print(f"\n  OVERALL: {'✅ ALL TESTS CONSISTENT WITH RH' if all_pass else '⚠️ SOME TESTS FAILED'}")
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
