# SWOTmodule_JAX

# Introduction
This repository contains a Python implementation of a denoising algorithm for Sea Surface Height (SSH) data, particularly from SWOT (Surface Water and Ocean Topography) satellite observations. The algorithm uses the LBFGS optimization method to minimize a cost function that includes Laplacian and third-order regularization terms. This method helps to remove noise from SSH images, improving data quality for further analysis.

# Requirements
To run the code, you need the following Python libraries:

- jax
- jax.numpy
- jaxopt
- pandas
- matplotlib
- netCDF4
- xarray

You can install these libraries using `pip`:


```pip install jax jaxopt pandas matplotlib netCDF4 xarray```

# Usage

**Import Packages**: The code begins with importing necessary packages for computation, data manipulation, and visualization.

**Reading Data:** The read_data function reads SSH data from a NetCDF file. This function requires the filename and the variables to be read as input arguments.

**Adding Noise:** The add_gaussian_noise function adds Gaussian noise to the original SSH image to simulate noisy observations.

**Computing Derivatives and Laplacian:**

- **derivative** function computes partial derivatives using a second-order centered scheme.
- **gradient** function computes the gradient of the input field.
- **laplacian** function computes the Laplacian of the input field.
- **third_order_terms** function calculates third-order terms inspired by the Quasi-Geostrophic model.

**Cost Functions:**
- **cost_function** calculates the cost function with Laplacian regularization.
- **cost_function_third_order_terms** calculates the cost function with both Laplacian and third-order regularization terms.

**RMSE Calculation:** The rmse function computes the Root Mean Square Error (RMSE) between two images.

**Plotting:** The **_splot_** function visualizes the SSH data, providing options to plot the
