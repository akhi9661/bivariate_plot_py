# Bivariate Plot

## Overview

The `bivariate_plot` function allows users to create bivariate scatter plots with color classification based on specified criteria. This visualization technique is useful for exploring the relationship between two continuous variables in a dataset and understanding how they interact. Similiar to `biscale` package in R.

## Features

- Supports classification of data using quantiles or equal bin sizes.
- Dynamically generates colors for the scatter points using various colormaps.
- Provides a customizable legend that reflects the data classification.
- Configurable plot aesthetics (labels, titles, sizes).

## Installation

To use the `bivariate_plot` function, you need to have Python and the required libraries installed. You can install the necessary packages via pip:

```bash
pip install pandas matplotlib
```

## Usage
### Importing libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

### Example
```python
# Sample Data
data = {
    'x': np.random.normal(50, 10, 2000),
    'y': np.random.normal(50, 10, 2000)
}

df = pd.DataFrame(data)

# Call the bivariate classification function
bivariate_plot(
    df, 
    'x', 
    'y', 
    style='quantile', 
    n_bins=4, 
    cmap_name='bwr', 
    alpha=0.7, 
    edgecolor='black',
    plot_kwargs={'labelsize': 14, 'title': 'Test Title'},
    legend_kwargs={'legend_position': (1.10, 0.15), 'legend_size': 0.2}
)
```

### Parameters

- `df`: `pandas.DataFrame`
  - The DataFrame containing the data to be plotted.

- `x`: `str`
  - The name of the column in `df` to be used for the x-axis.

- `y`: `str`
  - The name of the column in `df` to be used for the y-axis.

- `style`: `str`, optional
  - The method for classifying the data. Options are:
    - `'quantile'`: Classifies the data into quantiles.
    - `'equal'`: Classifies the data into equal-sized bins.
  - Default is `'quantile'`.

- `n_bins`: `int`, optional
  - The number of bins to use for classification.
  - Default is `3`.

- `cmap_name`: `str`, optional
  - The name of the colormap to use for coloring the points.
  - Default is `'viridis'`.

- `figsize`: `tuple`, optional
  - Size of the figure.
  - Default is `(8, 6)`.

- `legend_kwargs`: `dict`, optional
  - Arguments for customizing the legend (e.g., position, size).

- `plot_kwargs`: `dict`, optional
  - Arguments for customizing the main plot aesthetics (e.g., label size, title, etc.).

- `**plt_kwargs`: 
  - Additional keyword arguments passed to the `matplotlib` scatter function ( (e.g., color, edgecolor, etc.)).


## Contributing
If you'd like to contribute to this project, please fork the repository and submit a pull request with your changes. All contributions are welcome!

## License
This project is licensed under the MIT License. 

