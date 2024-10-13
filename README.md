[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akhi9661/bivariate_plot_py/blob/main/bivariate.ipynb)

# Bivariate Plot

## Overview

The `bivariate_plot_df.py` allows users to create bivariate scatter plots with color classification based on specified criteria. This visualization technique is useful for exploring the relationship between two continuous variables in a dataset and understanding how they interact. `bivariate_plot_raster.py` is for raster images similiar to `biscale` package in R.

## Features

- Supports classification of dataframe or raster using quantiles or equal interval.
- Configurable plot aesthetics (labels, titles, sizes).
- Option to overlay shapefiles for additional context on raster bivariate plot.
- Customizable colormaps and figure size.
- Customizable legend generation to help interpret the bivariate classifications.

## Installation

To use the function, you need to have Python and the required libraries installed. 
- Python 3.x
- numpy
- matplotlib
- rasterio (may require prior `gdal` installation too)
- geopandas
- os

You can install the necessary packages via pip:

```bash
pip install numpy pandas matplotlib rasterio geopandas
```

## Usage


### Dataframe bivariate scatter plot
```python
path = r'data\test.csv'
# Call the bivariate classification function using a continuous colormap
bivariate_plot_df(
    path, 'ADF Statistic', 'Keener Z-Statistic',
    style='quantile', 
    n_bins=5, 
    cmap_name='bwr', 
    alpha=0.7, 
    edgecolor='black', 
    plot_kwargs = {},
    legend_kwargs={'legend_position': (1.10, 0.15), 'legend_size': 0.2, 'ticklabelsize': 10}
)
```

![bivariate_df_plot](https://github.com/user-attachments/assets/686a22c8-a484-4cad-b75e-29a38554964e)


### Raster bivariate plot

```python
raster1_path = r'path/to/your/temperature_raster.TIF'
raster2_path = r'path/to/your/precipitation_raster.TIF'
shp_path = r'path/to/your/shapefile.shp'

bivariate_raster_plot(
    raster1_path, 
    raster2_path, 
    shp_path, 
    n_bins=5, 
    style='quantile', 
    cmap_name='coolwarm', 
    legend_kwargs={'ticklabelsize': 10, 'labelsize': 10, 'y_label': 'Precipitation (mm)', 'x_label':'Temperature (°C)'}
)

```

![bivariate_raster_plot](https://github.com/user-attachments/assets/673a11ba-71b2-449e-861a-609be037d7d8)


## Contributing
If you'd like to contribute to this project, please fork the repository and submit a pull request with your changes. All contributions are welcome!

## License
This project is licensed under the MIT License. 

