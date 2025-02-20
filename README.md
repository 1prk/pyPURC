# pyPURC

**pyPURC** is a Python implementation of the Perturbed Utility Route Choice Model as defined in Fosgerau, M., Paulsen, M., & Rasmussen, T. K. (2022). [A perturbed utility route choice model](https://www.sciencedirect.com/science/article/pii/S0968090X21004976). The goal of this project is to streamline the process so that it works out-of-the-box using OpenStreetMap data in conjunction with NetworkX, while also allowing for custom OD pairs provided with latitude and longitude coordinates.

## Proposed Features

- **OSM Integration:** Generate matrices from OpenStreetMap networks
- **Custom OD Pairs:** Aggregate OD pairs using H3 cells provided as latitude/longitude coordinates.
- and more..

## Installation

Ensure you have Python 3.12+ installed. The required packages are:
- `networkx`
- `numpy`
- `scipy`
- Other dependencies for handling OpenStreetMap data (e.g., `osmnx`) as needed.

You can install the required packages using pip:

```bash
pip install networkx numpy scipy osmnx
