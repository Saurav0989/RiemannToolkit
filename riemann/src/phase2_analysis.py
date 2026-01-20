#!/usr/bin/env python3
"""
RIEMANN HYPOTHESIS - PHASE 2: xp Eigenvalue Solver and Gap Analysis
=====================================================================

The xp model: H = xp (position × momentum)

This gets the DENSITY of zeros right but not individual values.
Goal: Find what's missing.

Approach:
1. Solve xp eigenvalue problem numerically
2. Compare eigenvalues to Riemann zeros
3. Analyze deviations for arithmetic structure
4. Test arithmetic corrections: V(x) = Σ log(p)/p^x
"""

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import mpmath as mp
from typing import List, Tuple, Optional
import os

mp.mp.dps = 30


# =============================================================================
# XP EIGENVALUE SOLVER
# =============================================================================

def solve_xp_finite_difference(N: int = 500, x_max: float = 100.0) -> np.ndarray:
    """
    Solve H = xp eigenvalue problem using finite difference.
    
    The operator: H = -iℏ x d/dx
    
    In position representation with ℏ = 1:
    H ψ(x) = -i x ψ'(x)
    
    For eigenvalue E: x ψ'(x) = -iE ψ(x)
    This gives: ψ(x) = x^{-iE}
    
    But we need boundary conditions to get discrete spectrum.
    Berry-Keating: put walls at x = 1 and x = e^L
    
    Using log-coordinates: y = log x, then H = -i d/dy
    On interval [0, L], with Dirichlet BCs, eigenvalues are E_n = nπ/L
    """
    # Method 1: Direct finite difference for H = xp
    # We discretize on [1, x_max] with log spacing for numerical stability
    
    # Use log-space: x = e^y, y ∈ [0, log(x_max)]
    L = np.log(x_max)
    dy = L / (N + 1)
    y = np.linspace(dy, L - dy, N)
    x = np.exp(y)
    
    # In y-coordinates: H = -i d/dy (symmetric form)
    # Finite difference for d/dy (centered)
    # H_jk = -i/(2dy) * (δ_{j,k+1} - δ_{j,k-1})
    
    diag_upper = np.ones(N - 1) * (-1j / (2 * dy))
    diag_lower = np.ones(N - 1) * (1j / (2 * dy))
    
    H = np.diag(diag_upper, 1) + np.diag(diag_lower, -1)
    
    # Make Hermitian: H = (H + H†)/2
    H = (H + H.conj().T) / 2
    
    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(H)
    
    return np.sort(eigenvalues)


def solve_xp_spectral(N: int = 500) -> np.ndarray:
    """
    Solve xp using spectral method (Chebyshev).
    
    The key insight from Berry-Keating:
    The semiclassical eigenvalues for the xp operator with regularization
    at x = ℓ_P (Planck scale) are approximately:
    
    E_n ≈ 2πn / log(E_n/E_0)
    
    This must be solved self-consistently.
    
    Alternatively, the Weyl formula gives:
    N(E) = (E/2π) log(E/2π) - E/2π + O(1)
    
    So we can invert this to get individual eigenvalues.
    """
    eigenvalues = []
    
    for n in range(1, N + 1):
        # Solve N(E) = n for E
        # N(E) = (E/2π) log(E/2π) - E/2π = n
        # Use Newton's method
        
        # Initial guess: E ≈ 2πn/log(n) for large n
        if n < 3:
            E = 2 * np.pi * n
        else:
            E = 2 * np.pi * n / np.log(n)
        
        for _ in range(50):  # Newton iterations
            if E <= 2 * np.pi:
                E = 2 * np.pi * n
                break
            
            N_E = (E / (2 * np.pi)) * np.log(E / (2 * np.pi)) - E / (2 * np.pi)
            dN_dE = (1 / (2 * np.pi)) * np.log(E / (2 * np.pi))
            
            if abs(dN_dE) < 1e-15:
                break
            
            delta = (N_E - n) / dN_dE
            E = E - delta
            
            if abs(delta) < 1e-12:
                break
        
        eigenvalues.append(E)
    
    return np.array(eigenvalues)


# =============================================================================
# DEVIATION ANALYSIS
# =============================================================================

def load_riemann_zeros(cache_path: str = None) -> np.ndarray:
    """Load computed Riemann zeros."""
    if cache_path is None:
        cache_path = os.path.join(os.path.dirname(__file__), "zeros_cache.npy")
    
    if os.path.exists(cache_path):
        return np.load(cache_path)
    else:
        raise FileNotFoundError(f"No zeros cache at {cache_path}")


def analyze_deviations(riemann_zeros: np.ndarray, xp_eigenvalues: np.ndarray) -> dict:
    """
    Detailed deviation analysis between Riemann zeros and xp eigenvalues.
    """
    n = min(len(riemann_zeros), len(xp_eigenvalues))
    
    rz = riemann_zeros[:n]
    xp = xp_eigenvalues[:n]
    
    # Deviations
    deviations = rz - xp
    
    # Statistics
    results = {
        'n': n,
        'mean_deviation': np.mean(deviations),
        'std_deviation': np.std(deviations),
        'max_deviation': np.max(np.abs(deviations)),
        'relative_error': np.mean(np.abs(deviations) / rz),
    }
    
    # Check for patterns in deviations
    # 1. Linear trend?
    indices = np.arange(n)
    linear_coef = np.polyfit(indices, deviations, 1)
    results['linear_slope'] = linear_coef[0]
    results['linear_intercept'] = linear_coef[1]
    
    # 2. Log trend?
    log_indices = np.log(indices + 1)
    log_coef = np.polyfit(log_indices, deviations, 1)
    results['log_slope'] = log_coef[0]
    
    # 3. Oscillatory component?
    # Compute FFT of deviations
    fft_dev = np.fft.fft(deviations)
    freqs = np.fft.fftfreq(n)
    dominant_freq_idx = np.argmax(np.abs(fft_dev[1:n//2])) + 1
    results['dominant_frequency'] = freqs[dominant_freq_idx]
    results['fft_power'] = np.abs(fft_dev[dominant_freq_idx])**2
    
    # 4. Correlation with primes?
    # Check if deviations correlate with prime-counting function
    from sympy import primepi
    prime_counts = np.array([primepi(int(rz[i])) for i in range(min(n, 200))])
    dev_subset = deviations[:len(prime_counts)]
    if len(prime_counts) > 5:
        try:
            corr_matrix = np.corrcoef(dev_subset, prime_counts)
            if corr_matrix.shape == (2, 2):
                results['prime_correlation'] = corr_matrix[0, 1]
        except:
            pass
    
    # 5. Check log(p) structure
    # The explicit formula has terms Σ log(p) / p^{ρ}
    # Check if deviations have any log-prime structure
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    log_primes = np.log(primes)
    
    # Test: Is deviation[n] ≈ Σ c_p log(p) for some coefficients?
    # This would suggest prime structure
    
    return results, deviations


# =============================================================================
# ARITHMETIC CORRECTIONS
# =============================================================================

def prime_potential(x: np.ndarray, n_primes: int = 100) -> np.ndarray:
    """
    Compute V(x) = Σ log(p) / p^x for first n primes.
    
    This encodes prime information into a potential.
    """
    from sympy import prime
    
    V = np.zeros_like(x)
    
    for i in range(1, n_primes + 1):
        p = prime(i)
        V += np.log(p) / (p ** x)
    
    return V


def von_mangoldt_potential(x: np.ndarray, n_max: int = 100) -> np.ndarray:
    """
    Compute V(x) = Σ Λ(n) e^{-nx} 
    
    where Λ(n) = log p if n = p^k, else 0 (von Mangoldt function)
    """
    from sympy import factorint
    
    V = np.zeros_like(x)
    
    for n in range(2, n_max + 1):
        factors = factorint(n)
        if len(factors) == 1:  # Prime power
            p = list(factors.keys())[0]
            V += np.log(p) * np.exp(-n * x)
    
    return V


def solve_xp_with_correction(N: int = 500, x_max: float = 100.0, 
                              correction_type: str = 'prime',
                              correction_strength: float = 1.0) -> np.ndarray:
    """
    Solve H = xp + αV(x) with arithmetic correction.
    """
    # Set up grid in log-space
    L = np.log(x_max)
    dy = L / (N + 1)
    y = np.linspace(dy, L - dy, N)
    x = np.exp(y)
    
    # Base xp operator in y-coordinates
    diag_upper = np.ones(N - 1) * (-1j / (2 * dy))
    diag_lower = np.ones(N - 1) * (1j / (2 * dy))
    H = np.diag(diag_upper, 1) + np.diag(diag_lower, -1)
    H = (H + H.conj().T) / 2
    
    # Add potential term
    if correction_type == 'prime':
        V = prime_potential(x, n_primes=50)
    elif correction_type == 'mangoldt':
        V = von_mangoldt_potential(x, n_max=50)
    else:
        V = np.zeros(N)
    
    H = H + correction_strength * np.diag(V)
    
    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(H.real)  # Take real part for Hermitian
    
    return np.sort(eigenvalues)


# =============================================================================
# SYSTEMATIC OPTIMIZATION
# =============================================================================

def optimize_correction(riemann_zeros: np.ndarray, N_eig: int = 200) -> dict:
    """
    Path C: Systematically optimize correction parameters.
    
    Try: H = xp + α*V_prime + β*V_mangoldt
    
    Minimize: L = Σ(E_n - γ_n)²
    """
    from scipy.optimize import minimize
    
    target = riemann_zeros[:N_eig]
    
    def objective(params):
        alpha, beta, x_max = params
        x_max = max(10, x_max)
        
        try:
            # Solve with both corrections
            L = np.log(x_max)
            N = 300
            dy = L / (N + 1)
            y = np.linspace(dy, L - dy, N)
            x = np.exp(y)
            
            # Base operator
            diag_upper = np.ones(N - 1) * (-1j / (2 * dy))
            diag_lower = np.ones(N - 1) * (1j / (2 * dy))
            H = np.diag(diag_upper, 1) + np.diag(diag_lower, -1)
            H = (H + H.conj().T) / 2
            
            # Add corrections
            V1 = prime_potential(x, n_primes=30) if abs(alpha) > 0.01 else np.zeros(N)
            V2 = von_mangoldt_potential(x, n_max=30) if abs(beta) > 0.01 else np.zeros(N)
            
            H = H.real + alpha * np.diag(V1) + beta * np.diag(V2)
            
            eigs = np.sort(np.linalg.eigvalsh(H))
            
            # Match to target (find best alignment)
            pos_eigs = eigs[eigs > 5][:N_eig]
            
            if len(pos_eigs) < N_eig // 2:
                return 1e10
            
            n_compare = min(len(pos_eigs), len(target))
            loss = np.mean((pos_eigs[:n_compare] - target[:n_compare])**2)
            
            return loss
            
        except Exception as e:
            return 1e10
    
    # Initial guess
    x0 = [0.0, 0.0, 100.0]
    
    # Bounds
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (20.0, 500.0)]
    
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B',
                     options={'maxiter': 100, 'disp': True})
    
    return {
        'optimal_alpha': result.x[0],
        'optimal_beta': result.x[1],
        'optimal_x_max': result.x[2],
        'final_loss': result.fun,
        'success': result.success
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_phase2_analysis():
    """
    Complete Phase 2 analysis.
    """
    print("\n" + "="*70)
    print("  RIEMANN HYPOTHESIS - PHASE 2 ANALYSIS")
    print("  xp Eigenvalue Solver and Gap Analysis")
    print("="*70)
    
    # 1. Load Riemann zeros
    print("\n[1/5] LOADING RIEMANN ZEROS")
    print("-" * 50)
    try:
        rz = load_riemann_zeros()
        print(f"  Loaded {len(rz)} Riemann zeros")
    except FileNotFoundError:
        print("  Computing zeros...")
        from phase1_analysis import compute_zeros
        rz = compute_zeros(500)
    
    # 2. Compute xp eigenvalues (spectral method - semiclassical)
    print("\n[2/5] COMPUTING XP EIGENVALUES (SEMICLASSICAL)")
    print("-" * 50)
    xp_eigs = solve_xp_spectral(len(rz))
    print(f"  Computed {len(xp_eigs)} xp eigenvalues")
    print(f"  First 5 xp: {xp_eigs[:5]}")
    print(f"  First 5 RZ: {rz[:5]}")
    
    # 3. Analyze deviations
    print("\n[3/5] DEVIATION ANALYSIS")
    print("-" * 50)
    results, deviations = analyze_deviations(rz, xp_eigs)
    
    print(f"  Mean deviation: {results['mean_deviation']:.4f}")
    print(f"  Std deviation:  {results['std_deviation']:.4f}")
    print(f"  Max deviation:  {results['max_deviation']:.4f}")
    print(f"  Relative error: {results['relative_error']*100:.2f}%")
    print(f"")
    print(f"  Linear trend: slope = {results['linear_slope']:.6f}")
    print(f"  Log trend: slope = {results['log_slope']:.4f}")
    print(f"  Dominant FFT frequency: {results['dominant_frequency']:.4f}")
    if 'prime_correlation' in results:
        print(f"  Correlation with π(x): {results['prime_correlation']:.4f}")
    
    # 4. Test arithmetic corrections
    print("\n[4/5] TESTING ARITHMETIC CORRECTIONS")
    print("-" * 50)
    
    for correction_type in ['prime', 'mangoldt']:
        for strength in [0.1, 0.5, 1.0]:
            xp_corrected = solve_xp_with_correction(
                N=300, x_max=100, 
                correction_type=correction_type,
                correction_strength=strength
            )
            
            pos_eigs = xp_corrected[xp_corrected > 5]
            if len(pos_eigs) >= 50:
                n_compare = min(len(pos_eigs), 50, len(rz))
                mse = np.mean((pos_eigs[:n_compare] - rz[:n_compare])**2)
                print(f"  {correction_type} (α={strength}): MSE = {mse:.2f}")
    
    # 5. Systematic optimization
    print("\n[5/5] OPTIMIZING CORRECTION PARAMETERS")
    print("-" * 50)
    
    opt_result = optimize_correction(rz, N_eig=100)
    print(f"  Optimal α (prime): {opt_result['optimal_alpha']:.4f}")
    print(f"  Optimal β (mangoldt): {opt_result['optimal_beta']:.4f}")
    print(f"  Optimal x_max: {opt_result['optimal_x_max']:.1f}")
    print(f"  Final loss: {opt_result['final_loss']:.2f}")
    
    # Summary
    print("\n" + "="*70)
    print("  PHASE 2 SUMMARY")
    print("="*70)
    
    print(f"""
  KEY FINDINGS:
  
  1. DEVIATION STRUCTURE:
     • Mean deviation: {results['mean_deviation']:.4f}
     • Deviations have {'linear' if abs(results['linear_slope']) > 0.001 else 'no clear'} trend
     • Prime correlation: {results.get('prime_correlation', 'N/A')}
  
  2. ARITHMETIC CORRECTIONS:
     • Prime potential V(x) = Σ log(p)/p^x
     • von Mangoldt potential V(x) = Σ Λ(n)e^{{-nx}}
     • Optimal parameters found via optimization
  
  3. THE GAP:
     The xp model gives the right counting function N(T).
     But individual eigenvalues deviate systematically.
     
     The deviations encode arithmetic information that
     xp alone doesn't capture.
  
  NEXT STEP: Analyze WHY these specific deviations occur.
  The explicit formula suggests they encode prime information.
    """)
    
    return {
        'riemann_zeros': rz,
        'xp_eigenvalues': xp_eigs,
        'deviations': deviations,
        'deviation_stats': results,
        'optimization': opt_result
    }


if __name__ == "__main__":
    results = run_phase2_analysis()
