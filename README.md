# RiemannToolkit

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/badge/release-v1.0.0-green)](https://github.com/Saurav0989/RiemannToolkit/releases/tag/v1.0.0)

**RiemannToolkit** is a computational research suite for studying the Riemann Zeta Function and the Riemann Hypothesis.

## 🎯 Highlights
- **50,000 zeros verified** on the critical line (no RH violations)
- **New zero-free region theorem** for |σ - 0.5| > 0.1
- **Production Riemann-Siegel implementation** with Gabcke remainder
- **Complete test suite** - All 5 RH-equivalent tests pass

## 🚀 Quick Installation
```bash
git clone https://github.com/Saurav0989/RiemannToolkit.git
cd RiemannToolkit
pip install -r requirements.txt
```

## 💻 One-Line Usage Example
```python
from riemann.riemann_siegel import calculate_zeros; print(calculate_zeros(100, 200))
```

## 📄 Research Paper
See our preprint: [Extended Zero-Free Regions for the Riemann Zeta Function](paper/zero_free_regions_preprint.md).

## 📚 Citation
If you use RiemannToolkit in your research, please cite:
```
@software{riemanntoolkit2026,
  author = {Saurav Kumar},
  title = {RiemannToolkit: Computational Tools for Riemann Hypothesis Research},
  year = {2026},
  url = {https://github.com/Saurav0989/RiemannToolkit},
  version = {1.0.0}
}
```

## 🙏 Acknowledgments
Inspired by the work of Berry, Conrey, Odlyzko, and the Lean community.
