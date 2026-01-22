# RiemannToolkit: High-Precision Visualization of the Zeta Critical Line

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status: Active](https://img.shields.io/badge/Status-Active-green.svg)]()

**RiemannToolkit** is an open-source Python library designed for the numerical analysis and visualization of the Riemann Zeta function along the critical line ($\sigma = 0.5$).

Unlike symbolic solvers, this toolkit focuses on **computational observability**—providing researchers with a granular view of zero-crossings, error margins, and asymptotic behaviors using high-precision libraries (`mpmath`).

## 🚀 Key Features

* **Critical Line Scanning:** Automated detection of non-trivial zeros within a specified range ($t_{min}, t_{max}$).
* **Error Analysis:** Quantitative visualization of the $Z(t)$ function (Hardy Z-function) deviations.
* **Fast Rendering:** Optimized plotting pipeline for generating high-resolution critical strip visualizations.
* **Interactive Notebooks:** Ready-to-use Jupyter notebooks for exploring prime number distribution correlations.

## 🛠️ Installation

```bash
git clone [https://github.com/Saurav0989/RiemannToolkit.git](https://github.com/Saurav0989/RiemannToolkit.git)
cd RiemannToolkit
pip install -r requirements.txt

📊 Usage Example
Visualizing the first 5 non-trivial zeros:
from riemann_toolkit import ZetaScanner, Visualizer

# Initialize scanner for the critical line
scanner = ZetaScanner(precision=50)

# Scan range t=[0, 30]
zeros = scanner.find_zeros(start=0, end=30)
print(f"Found {len(zeros)} zeros: {zeros}")

# Generate Plot
Visualizer.plot_critical_line(range=(0, 30), show_zeros=True)

🔬 Scientific Motivation
This project aims to provide an independent computational verification framework for investigating the local properties of the Zeta function. It serves as a workbench for testing conjectures regarding the spacing distribution of zeros (Montgomery's Pair Correlation Conjecture).

🤝 Contributing
This is an active research tool. Pull requests for optimizing the Gram point calculation or extending the visualization to the critical strip (0<σ<1) are welcome.

📜 License
MIT License - Free for academic and research use.
