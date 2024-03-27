# Fast-DCP

Python implementation of Ding J-J, Tu M-Y, Hua S-C. Fast Algorithms for Multi-Windowed Dark Channel Priors with Less Comparison and Time Cycle Requirement. IET ICETA 2023.

## Requirements

- numpy==1.26.4

## Usage

```python
from fastDCP import fastDCP
dcps = fastDCP(image, layer)
```

## Dehazing Results

### Requirements

- opencv-python==4.9.0.80
- opencv-contrib-python==4.9.0.80

### Usage

```bash
cd dcp-dehaze
python main.py
```
