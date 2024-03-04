## Credits
The MMAR class is inpired by the work of Federico Maglione: ["Fractal and Multifractal models for price changes"](https://www.academia.edu/89617926/Fractal_and_Multifractal_models_for_price_changes), even if the original paper used R. The code has been freely translated into Python, partly using ChatGPT that provided a convenient way to transalte R functions' calls into Python ones. The generated translation was far from perfect, but still a time saving. The code has then been refactored to be more *Pythonic* and tested. In some cases we have made different choices, for example in the selection of the scaling factors, which is why the two models can generate different results.

The computation of $\theta$ using Volumes is taken from a paper by Batten, Kinateder, and Wagner: ["Multifractality and value-at-risk forecasting of exchange rate"](https://doi.org/10.1016/j.physa.2014.01.024)

## Notes
The class uses **Numba** to speed up the Monte Carlo Simulation, which remains a bottleneck in some situations. We aim to provide a GPU accelerated method in the future versions. Memory is another important issue: the array to store a million simulations with 2,048 steps using `float64`, is around 16 GB! Memory use can be optimized, but in this case it becomes difficult to guarantee reproducibility given the stochastic nature of the process. 

## Authors:
- Andrea Dalseno
- Vsevolod Cherepanov
- Sanchit Jain