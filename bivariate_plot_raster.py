import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import matplotlib.patches as mpatches
import geopandas as gpd
from matplotlib.colors import ListedColormap
import os

def classify_raster(data, n_bins=5, style='quantile'):
    """
    Classify raster data using quantile or equal intervals, masking NaN values.

    Parameters:
    ----------
    data : np.ndarray
        The raster data as a 2D numpy array.
    n_bins : int, optional
        The number of bins for classification. Default is 5.
    style : str, optional
        The classification method. Options are 'quantile' or 'equal'. Default is 'quantile'.

    Returns:
    -------
    classified_data : np.ndarray
        Classified data as a 2D numpy array with values from 1 to n_bins, with NaN values masked.
    bin_edges : list
        The edges of the bins used for classification, useful for legend labels.
    """
    # Replace non-finite values with np.nan
    data = np.where(np.isfinite(data), data, np.nan)

    # Mask out NaN values
    masked_data = np.ma.masked_invalid(data)

    if style == 'quantile':
        bin_edges = np.nanquantile(masked_data.compressed(), np.linspace(0, 1, n_bins + 1))
        classified_data = np.full(masked_data.shape, np.nan, dtype=float)  # Initialize as float
        classified_data[masked_data.mask] = np.nan  # Ensure NaN for masked areas
        classified_data[~masked_data.mask] = np.digitize(masked_data.compressed(), bin_edges[1:-1]) + 1
    elif style == 'equal':
        bin_edges = np.linspace(np.nanmin(masked_data), np.nanmax(masked_data), n_bins + 1)
        classified_data = np.full(masked_data.shape, np.nan, dtype=float)  # Initialize as float
        classified_data[masked_data.mask] = np.nan  # Ensure NaN for masked areas
        classified_data[~masked_data.mask] = np.digitize(masked_data.compressed(), bin_edges[1:-1]) + 1
    else:
        raise ValueError("Invalid style. Use 'quantile' or 'equal'.")

    return classified_data, bin_edges

def bivariate_raster_plot(raster1_path, raster2_path, shp_path=None, n_bins=5, style='quantile', cmap_name='viridis', 
                           figsize=(10, 8), main_kwargs={}, legend_kwargs={}):
    """
    Create a bivariate raster plot with color classification.

    Parameters:
    ----------
    raster1_path : str
        The file path to the first raster data (e.g., temperature).
    raster2_path : str
        The file path to the second raster data (e.g., precipitation).
    shp_path : str, optional
        The file path to a shapefile to overlay on the plot. Default is None.
    n_bins : int, optional
        The number of bins for classification. Default is 5.
    style : str, optional
        The classification method ('quantile' or 'equal'). Default is 'quantile'.
    cmap_name : str, optional
        The colormap name. Default is 'viridis'.
    figsize : tuple, optional
        The size of the figure. Default is (10, 8).
    main_kwargs : dict, optional
        Additional arguments for the main plot (title, axis labels).
    legend_kwargs : dict, optional
        Additional arguments for the legend (label sizes, axis labels).

    Returns:
    -------
    None
        Displays the bivariate raster plot.
    """
    
    ticklabelsize = main_kwargs.get('ticklabelsize', None)
    labelsize = main_kwargs.get('labelsize', 10)
    x_label = main_kwargs.get('x_label', '')
    y_label = main_kwargs.get('y_label', '')
    title = main_kwargs.get('title', '')
    titlesize = main_kwargs.get('titlesize', 14)

    # Read the first raster and its metadata
    with rasterio.open(raster1_path) as src1:
        raster1 = src1.read(1)  # Read the first band
        height, width = raster1.shape  # Get dimensions
        transform=src1.transform
        crs=src1.crs

    with rasterio.open(raster2_path) as src2:
        raster2 = src2.read(1)  # Read the first band
    
    # Step 1: Classify both raster variables and get bin edges for legend
    classified_raster1, bin_edges1 = classify_raster(raster1, n_bins=n_bins, style=style)
    classified_raster2, bin_edges2 = classify_raster(raster2, n_bins=n_bins, style=style)

    # Step 2: Create bivariate classification
    bivar_class = (classified_raster1 - 1) * n_bins + (classified_raster2 - 1)  # Bivariate index (0 to n_bins**2 - 1)

    # Set NaN values in bivar_class to NaN
    bivar_class[np.isnan(classified_raster1) | np.isnan(classified_raster2)] = np.nan

    # Step 3: Generate colors based on the bivariate classification
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, n_bins ** 2))  # Generate colors for each class
    
    # Create an empty RGBA image
    color_image = np.empty((height, width, 4), dtype=np.float32)
    color_image[:] = np.nan  # Set the entire grid to NaN
    
    # Assign colors for valid classifications
    valid_indices = np.isfinite(bivar_class)
    color_image[valid_indices] = colors[bivar_class[valid_indices].astype(int)]  # Map classification to colors

    fig, ax = plt.subplots(figsize=figsize)
    # Calculate the extent (bounding box) from the affine transform for georeferencing
    extent = (
        transform[2],  # left (x min)
        transform[2] + transform[0] * width,  # right (x max)
        transform[5] + transform[4] * height,  # bottom (y min)
        transform[5]  # top (y max)
    )
    
    # Plot the classified raster using imshow() with extent to preserve geospatial info
    ax.imshow(color_image, extent=extent, origin='upper')  

    # Plot the shapefile if provided
    if shp_path is not None:
        gdf = gpd.read_file(shp_path)
        gdf = gdf.to_crs(crs)  # Ensure the shapefile is in the same CRS as the raster
        gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.2)  # Overlay shapefile

    # Add title and labels
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(x_label, fontsize=labelsize)
    ax.set_ylabel(y_label, fontsize=labelsize)

    if ticklabelsize is None:
        plt.xticks([])
        plt.yticks([])
    
    # Step 5: Create the bivariate legend using actual bin edges
    create_bivariate_legend(fig, ax, n_bins, colors, bin_edges1, bin_edges2, 
                            **legend_kwargs)
    
    plt.show()
    
    #os.remove(output_tiff_path)

def create_bivariate_legend(fig, ax, n_bins, colors, bin_edges1, bin_edges2, 
                             **legend_kwargs):
    """
    Create a custom square legend for the bivariate plot.

    Parameters:
    ----------
    fig : matplotlib.figure.Figure
        The figure object to which the legend will be added.
    ax : matplotlib.axes.Axes
        The axes to draw the legend on.
    n_bins : int
        The number of bins for classification.
    colors : list
        The list of colors for the legend.
    bin_edges1 : list
        The bin edges for the first raster.
    bin_edges2 : list
        The bin edges for the second raster.
    
    Returns:
    -------
    None
        Adds a legend to the plot.
    """
    legend_size = legend_kwargs.get('legend_size', 0.2)
    ticklabelsize = legend_kwargs.get('ticklabelsize', 10)
    labelsize = legend_kwargs.get('labelsize', 12)
    x_label = legend_kwargs.get('x_label', 'Raster 1')
    y_label = legend_kwargs.get('y_label', 'Raster 2')
    
    # Reshape colors into an n_bins x n_bins grid
    color_grid = np.array(colors).reshape((n_bins, n_bins, -1))

    # Create legend as a new axis
    bbox = ax.get_position()
    legend_ax = fig.add_axes([bbox.x1 + 0.08, bbox.y0 + 0.06, legend_size, legend_size])

    for i in range(n_bins):
        for j in range(n_bins):
            legend_ax.add_patch(mpatches.Rectangle((j, i), 1, 1, color=color_grid[i, j]))

    legend_ax.set_xlim(0, n_bins)
    legend_ax.set_ylim(0, n_bins)
    legend_ax.set_xticks(np.arange(n_bins) + 0.5)
    #legend_ax.set_xticklabels([f'{int(bin_edges1[i])}' for i in range(n_bins)], fontsize=ticklabelsize)  # Show 0 decimal points
    legend_ax.set_xticklabels([f'{bin_edges1[i]:.1f}' for i in range(n_bins)], fontsize=ticklabelsize)  # Show 0 decimal points
    
    legend_ax.set_yticks(np.arange(n_bins) + 0.5)
    legend_ax.set_yticklabels([f'{bin_edges2[i]:.1f}' for i in range(n_bins)], fontsize=ticklabelsize)  # Show 0 decimal points
    
    legend_ax.set_xlabel(x_label, fontsize=labelsize)
    legend_ax.set_ylabel(y_label, fontsize=labelsize)
