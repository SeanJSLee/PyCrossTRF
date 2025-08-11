# PyCrossTRF

PyCrossTRF implements cross‑sectional transfer‑function (TRF) and covariate‑augmented transfer‑function (CTRF) models for analyzing how outcomes (e.g., mortality, yields) respond to temperature across many regions. The package follows the approach of Chang et al. (2016) and is designed for applied economists working with panel climate–impact data.

## Key Features
- **TRF/CTRF estimation** – CTRF builds polynomial–Fourier bases and fits OLS to recover the response curve and the minimum mortality temperature (MMT)
- **Covariate handling** – covariates are centered/standardized and interact with temperature to produce CTRFs
- **Sieve bootstrap** – `SieveBootstrap` generates residual‑based confidence bands via AR‑sieve resampling
- **Visualization utilities** – `plotting.Plot` overlays TRFs/CTRFs and diagnostics for publication‑quality figures

## Installation
```bash
pip install PyCrossTRF        # from PyPI (if available)
# or install locally
pip install -e .
```

## Quick Start

### 1. Prepare data
Your DataFrame must include:

| Column        | Description |
|---------------|-------------|
| dep           | dependent variable (e.g., log mortality) |
| temp_q        | temperature regressor |
| date          | time index |
| fips          | cross‑sectional identifier |
| other covariates | optional (e.g., trends, demographics) |

### 2. Fit a TRF
```python
import PyCrossTRF as ctrf

pq = {'p': 4, 'q': 2}  # polynomial/Fourier orders
model = ctrf.CTRF(df=df,
                  dep='ln_mortality',
                  temp_r='temp_q',
                  covariates=[],
                  pq_order=pq,
                  scale={'temp_q': 'raw'})   # normalization rule
model.estimate_trf()
ctrf.plotting.Plot(model).trf_plot()
```

### 3. Add Covariates (CTRF)
```python
model = ctrf.CTRF(df=df,
                  dep='ln_mortality',
                  temp_r='temp_q',
                  covariates=['time_trend'],
                  pq_order=pq,
                  scale={'temp_q': 'raw', 'time_trend': 'linear'})
model.estimate_ctrf()
ctrf.plotting.Plot(model).ctrf_plot()
```

### 4. Bootstrap Confidence Bands
```python
bs  = ctrf.SieveBootstrap(model, maxL=24, B=999)
boot = bs.run()

lb = ctrf.CTRF(df=boot, dep='y_star_tau025', temp_r='temp_q',
               covariates=[], pq_order=pq, scale={'temp_q':'raw'})
lb.estimate_trf()

ub = ctrf.CTRF(df=boot, dep='y_star_tau975', temp_r='temp_q',
               covariates=[], pq_order=pq, scale={'temp_q':'raw'})
ub.estimate_trf()

ctrf.plotting.Plot(ub).trf_plot()  # overlay bounds
```

## Interpreting Results
- **TRF curve** – shows expected outcome for each normalized temperature quantile.
- **MMT** – the temperature quantile minimizing the outcome is calculated automatically.
- **CTRF curves** – shifts in the response curve under different covariate values illustrate heterogeneous effects.
- **Bootstrap bands** – 2.5%/97.5% envelopes for policy‑relevant uncertainty assessment.

## Citation
If you use this package, please cite Chang et al. (2016) and this repository.

## License
MIT © 2024 Jaeseok Sean Lee
