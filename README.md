# Option-Pricing-Model

## Does
* Run the compiled executable to forecast the option's price at maturity and cross-references with market option price to find largest arbitrage opportunity
* Uses Geometric Brownian Motion (GBM) and Jump Diffusion stochastic progresses to model the distribution of returns for a stock, then randomly pulls a daily return values for each time step of the simulation and computes the average return of all run simulations

## Usage

```bash
./executable `path-to-csv`
```

## Notes
* CSV file must be formatted according to ***Yahoo Finance's*** Historical Datasets
