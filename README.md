# power_spectrum_estimator_py
A Python package for computing matter power spectra from cosmological N-body simulations. Supports multiple mass assignment schemes and Fourier transform methods.

## Features

- **Mass Assignment Schemes**: NGP | CIC | TSC
- **Anti-Aliasing**: Interlaced grid technique
- **Window Correction**: Deconvolution in Fourier space


## Test
```
pytest -s ./test/test.py
```

