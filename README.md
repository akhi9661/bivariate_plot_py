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

You can install the necessary packages via pip:

```bash
pip install numpy pandas matplotlib rasterio geopandas
```

## Usage


### Dataframe bivariate scatter plot
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

## Contributing
If you'd like to contribute to this project, please fork the repository and submit a pull request with your changes. All contributions are welcome!

## License
This project is licensed under the MIT License. 

