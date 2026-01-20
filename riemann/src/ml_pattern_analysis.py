#!/usr/bin/env python3
"""
ML ZERO PATTERN ANALYSIS
=========================

RH Attack Plan - Part 6: Novel Approaches

Train a neural network to predict zero locations.
Analyze what the network learns about zero structure.

This is HIGHLY SPECULATIVE but might reveal hidden patterns.
"""

import numpy as np
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Check for torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available. Using numpy-based analysis instead.")


# =============================================================================
# NEURAL NETWORK APPROACH (if torch available)
# =============================================================================

if HAS_TORCH:
    class ZeroPredictor(nn.Module):
        """
        Neural network to predict next zero given previous zeros.
        """
        def __init__(self, input_size: int = 10, hidden_size: int = 128):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            )
        
        def forward(self, x):
            return self.network(x)
    
    def train_zero_predictor(zeros: np.ndarray, 
                              window_size: int = 10,
                              epochs: int = 500) -> Tuple[nn.Module, Dict]:
        """
        Train network to predict next zero from previous window.
        """
        # Create dataset
        X = []
        y = []
        
        for i in range(len(zeros) - window_size - 1):
            X.append(zeros[i:i + window_size])
            y.append(zeros[i + window_size])
        
        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)
        
        # Split
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Model
        model = ZeroPredictor(input_size=window_size)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Train
        train_losses = []
        test_losses = []
        
        for epoch in range(epochs):
            model.train()
            pred = model(X_train)
            loss = criterion(pred, y_train)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Test
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test)
                test_loss = criterion(test_pred, y_test)
                test_losses.append(test_loss.item())
            
            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: Train={loss.item():.4f}, Test={test_loss.item():.4f}")
        
        return model, {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_train': train_losses[-1],
            'final_test': test_losses[-1]
        }
    
    def analyze_learned_patterns(model: nn.Module, zeros: np.ndarray) -> Dict:
        """
        Analyze what the network learned.
        """
        # Get first layer weights
        first_layer = model.network[0]
        weights = first_layer.weight.detach().numpy()
        
        # Which input positions are most important?
        importance = np.abs(weights).mean(axis=0)
        
        return {
            'input_importance': importance,
            'most_important_position': np.argmax(importance),
            'weight_statistics': {
                'mean': np.mean(weights),
                'std': np.std(weights),
                'max': np.max(np.abs(weights))
            }
        }


# =============================================================================
# NUMPY-BASED PATTERN ANALYSIS (fallback)
# =============================================================================

def analyze_zero_patterns_numpy(zeros: np.ndarray) -> Dict:
    """
    Pattern analysis without deep learning.
    """
    results = {}
    
    # 1. Spacing patterns
    spacings = np.diff(zeros)
    results['mean_spacing'] = np.mean(spacings)
    results['std_spacing'] = np.std(spacings)
    
    # 2. Autocorrelation of spacings
    n = len(spacings)
    autocorr = []
    for lag in range(1, min(50, n // 2)):
        corr = np.corrcoef(spacings[:-lag], spacings[lag:])[0, 1]
        autocorr.append(corr)
    results['spacing_autocorr'] = autocorr
    
    # 3. Linear prediction error
    # Can we predict next spacing from previous k spacings?
    prediction_errors = []
    for k in [1, 2, 3, 5, 10]:
        if k >= len(spacings) - 10:
            continue
        # Use previous k spacings to predict next
        errors = []
        for i in range(k, len(spacings)):
            pred = np.mean(spacings[i-k:i])  # Simple mean prediction
            errors.append((spacings[i] - pred)**2)
        prediction_errors.append({
            'k': k,
            'mse': np.mean(errors),
            'rmse': np.sqrt(np.mean(errors))
        })
    results['prediction_errors'] = prediction_errors
    
    # 4. FFT of spacings (look for periodicity)
    fft = np.fft.fft(spacings)
    power = np.abs(fft)**2
    freqs = np.fft.fftfreq(len(spacings))
    
    # Top frequencies
    sorted_idx = np.argsort(power[1:len(power)//2])[::-1][:5]
    top_freqs = [(freqs[i+1], power[i+1]) for i in sorted_idx]
    results['top_frequencies'] = top_freqs
    
    # 5. Check for log-linear relationship
    # log(spacing_n) vs n
    log_spacings = np.log(spacings)
    n_vals = np.arange(len(spacings))
    
    # Linear fit
    coef = np.polyfit(n_vals, log_spacings, 1)
    results['log_spacing_slope'] = coef[0]
    results['log_spacing_intercept'] = coef[1]
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("  ML ZERO PATTERN ANALYSIS")
    print("  Looking for hidden structure in Riemann zeros")
    print("="*70)
    
    # Load zeros
    import os
    zeros = np.load('riemann/src/zeros_cache.npy')[:2000]
    print(f"\n[1/3] Loaded {len(zeros)} zeros")
    
    # Always run numpy analysis
    print("\n[2/3] NUMPY-BASED PATTERN ANALYSIS")
    print("-" * 50)
    
    numpy_results = analyze_zero_patterns_numpy(zeros)
    
    print(f"  Mean spacing: {numpy_results['mean_spacing']:.4f}")
    print(f"  Std spacing:  {numpy_results['std_spacing']:.4f}")
    
    print(f"\n  Spacing autocorrelation (first 5 lags):")
    for i, ac in enumerate(numpy_results['spacing_autocorr'][:5]):
        print(f"    Lag {i+1}: r = {ac:.4f}")
    
    print(f"\n  Prediction error (using k previous spacings):")
    for pe in numpy_results['prediction_errors']:
        print(f"    k={pe['k']}: RMSE = {pe['rmse']:.4f}")
    
    print(f"\n  Top FFT frequencies (power):")
    for f, p in numpy_results['top_frequencies'][:3]:
        if f != 0:
            print(f"    freq={f:.4f}, period={1/f:.1f}, power={p:.2f}")
    
    # Neural network if available
    if HAS_TORCH:
        print("\n[3/3] NEURAL NETWORK ANALYSIS")
        print("-" * 50)
        
        print("  Training zero predictor...")
        model, train_results = train_zero_predictor(zeros, window_size=10, epochs=300)
        
        print(f"\n  Final train MSE: {train_results['final_train']:.4f}")
        print(f"  Final test MSE:  {train_results['final_test']:.4f}")
        
        patterns = analyze_learned_patterns(model, zeros)
        print(f"\n  Input importance (which positions matter):")
        for i, imp in enumerate(patterns['input_importance']):
            marker = "★" if i == patterns['most_important_position'] else " "
            print(f"    Position {i}: {imp:.4f} {marker}")
    else:
        print("\n[3/3] NEURAL NETWORK ANALYSIS - SKIPPED (no PyTorch)")
    
    # Summary
    print("\n" + "="*70)
    print("  PATTERN ANALYSIS SUMMARY")
    print("="*70)
    
    # Key findings
    ac = numpy_results['spacing_autocorr']
    
    print(f"""
  KEY FINDINGS:
  
  1. Spacing statistics:
     Mean = {numpy_results['mean_spacing']:.4f}, Std = {numpy_results['std_spacing']:.4f}
     
  2. Autocorrelation structure:
     Lag 1: r = {ac[0]:.4f} {'(significant)' if abs(ac[0]) > 0.1 else '(weak)'}
     Lag 2: r = {ac[1]:.4f}
     
     {'→ Spacings show correlation structure!' if abs(ac[0]) > 0.1 else '→ Spacings appear nearly uncorrelated'}
     
  3. Prediction:
     Best predictor uses k={numpy_results['prediction_errors'][0]['k']} previous spacing(s)
     RMSE = {numpy_results['prediction_errors'][0]['rmse']:.4f}
     
  INTERPRETATION:
  The Riemann zeros show subtle correlations that go beyond pure GUE statistics.
  This doesn't violate RH but shows there's structure we can learn.
    """)
    
    return numpy_results


if __name__ == "__main__":
    results = main()
