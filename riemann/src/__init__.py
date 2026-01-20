"""
RiemannToolkit: Computational Tools for Riemann Zeta Research
===============================================================

A comprehensive Python library for high-precision zeta function
zero computation, statistical analysis, and RH verification.
"""

from .riemann_siegel import (
    riemann_siegel_Z,
    riemann_siegel_theta,
    find_zeros_in_range,
    refine_zero,
    validate_implementation
)

from .mertens import (
    mobius_sieve,
    mertens_function,
    analyze_mertens,
    test_rh_bound
)

from .deep_study import (
    zeta_definition,
    functional_equation,
    explicit_formula,
    why_critical_line
)

__version__ = "0.1.0"
__author__ = "RH Attack Project"

__all__ = [
    # Riemann-Siegel
    'riemann_siegel_Z',
    'riemann_siegel_theta', 
    'find_zeros_in_range',
    'refine_zero',
    'validate_implementation',
    
    # Mertens
    'mobius_sieve',
    'mertens_function',
    'analyze_mertens',
    'test_rh_bound',
    
    # Educational
    'zeta_definition',
    'functional_equation',
    'explicit_formula',
    'why_critical_line'
]
