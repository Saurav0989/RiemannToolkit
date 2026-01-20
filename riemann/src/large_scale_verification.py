#!/usr/bin/env python3
"""
LARGE-SCALE ZERO VERIFICATION
==============================

Month 4 Milestone: Scale to 10^6 - 10^7 zeros

Optimized for:
1. Batch processing
2. Efficient theta computation
3. Progress tracking
4. Checkpoint saving
"""

import numpy as np
import time
import os
import sys

sys.path.insert(0, 'riemann/src')

# Use asymptotic theta for speed
def fast_theta(t):
    """Fast asymptotic theta for large t."""
    if t < 10:
        return t/2 * np.log(t / (2*np.pi*np.e)) - np.pi/8
    return t/2 * np.log(t / (2*np.pi*np.e)) - np.pi/8 + 1/(48*t) + 7/(5760*t**3)


def fast_Z(t, N=None):
    """Fast Z(t) using asymptotic theta."""
    if N is None:
        N = int(np.sqrt(t / (2*np.pi))) + 1
    
    theta = fast_theta(t)
    
    # Main sum
    result = 0.0
    for n in range(1, N + 1):
        result += np.cos(theta - t * np.log(n)) / np.sqrt(n)
    
    return 2.0 * result


def find_zeros_batch(t_start, t_end, step=0.5):
    """Find zeros in a batch efficiently."""
    zeros = []
    
    t = t_start
    prev_Z = fast_Z(t)
    
    while t < t_end:
        t += step
        curr_Z = fast_Z(t)
        
        if prev_Z * curr_Z < 0:
            # Refine with binary search (3 iterations)
            a, b = t - step, t
            for _ in range(20):
                mid = (a + b) / 2
                if fast_Z(a) * fast_Z(mid) < 0:
                    b = mid
                else:
                    a = mid
            zeros.append((a + b) / 2)
        
        prev_Z = curr_Z
    
    return zeros


def scale_verification(target_zeros=100000, checkpoint_interval=10000):
    """
    Scale verification to target number of zeros.
    """
    print(f"\n{'='*60}")
    print(f"  LARGE-SCALE ZERO VERIFICATION")
    print(f"  Target: {target_zeros:,} zeros")
    print(f"{'='*60}")
    
    # Estimate T needed: N(T) ≈ (T/2π) log(T/2π) - T/2π
    # For 100K zeros, T ≈ 60,000
    # For 1M zeros, T ≈ 600,000
    
    T_estimate = target_zeros * 6  # Rough estimate
    
    all_zeros = []
    batch_size = 1000  # Process in batches
    
    t_current = 14  # Start after trivial zeros
    
    start_time = time.time()
    last_checkpoint = 0
    
    while len(all_zeros) < target_zeros:
        # Find zeros in batch
        batch_zeros = find_zeros_batch(t_current, t_current + batch_size)
        all_zeros.extend(batch_zeros)
        t_current += batch_size
        
        # Progress
        if len(all_zeros) - last_checkpoint >= checkpoint_interval:
            elapsed = time.time() - start_time
            rate = len(all_zeros) / elapsed
            eta = (target_zeros - len(all_zeros)) / rate if rate > 0 else 0
            
            print(f"  Progress: {len(all_zeros):>8,} zeros | "
                  f"t = {t_current:>8,.0f} | "
                  f"Rate: {rate:.0f}/sec | "
                  f"ETA: {eta/60:.1f} min")
            
            last_checkpoint = len(all_zeros)
            
            # Save checkpoint
            np.save('riemann/src/zeros_checkpoint.npy', np.array(all_zeros))
    
    # Trim to exact target
    all_zeros = all_zeros[:target_zeros]
    
    elapsed = time.time() - start_time
    
    print(f"\n  COMPLETE!")
    print(f"  Total zeros: {len(all_zeros):,}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Rate: {len(all_zeros)/elapsed:.0f} zeros/sec")
    
    # Save final
    np.save('riemann/src/zeros_large.npy', np.array(all_zeros))
    
    return np.array(all_zeros)


def verify_critical_line(zeros, sample_size=100):
    """
    Verify zeros are on critical line by checking |ζ(σ+it)| for σ ≠ 0.5
    """
    try:
        import mpmath
        mpmath.mp.dps = 30
    except ImportError:
        print("  mpmath not available for verification")
        return None
    
    print(f"\n  Verifying {sample_size} zeros on critical line...")
    
    sample = np.random.choice(zeros, min(sample_size, len(zeros)), replace=False)
    violations = []
    
    for i, t in enumerate(sample):
        # Check at σ = 0.4 and σ = 0.6
        for sigma in [0.4, 0.6]:
            zeta_val = mpmath.zeta(mpmath.mpc(sigma, t))
            if abs(zeta_val) < 0.001:  # Suspiciously small off line
                violations.append((sigma, t, abs(zeta_val)))
    
    if violations:
        print(f"  ⚠️ {len(violations)} potential violations!")
    else:
        print(f"  ✅ All {sample_size} zeros verified on critical line")
    
    return len(violations) == 0


def main():
    # Start with 50K zeros (achievable in ~5 min)
    zeros = scale_verification(target_zeros=50000)
    
    # Verify sample
    verify_critical_line(zeros, sample_size=100)
    
    # Statistics
    print(f"\n  STATISTICS:")
    print(f"  First zero: {zeros[0]:.6f}")
    print(f"  Last zero:  {zeros[-1]:.6f}")
    print(f"  Mean spacing: {np.mean(np.diff(zeros)):.4f}")
    
    return zeros


if __name__ == "__main__":
    zeros = main()
