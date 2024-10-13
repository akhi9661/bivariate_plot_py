import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def bivariate_plot_df(df_path, x, y, style='quantile', n_bins=3, cmap_name='viridis', figsize=(8, 6), legend_kwargs=None, plot_kwargs=None, **plt_kwargs):
    """
    Create a bivariate scatter plot with color classification based on specified criteria.
    
    Parameters:
    df_path : str
        The path to the csv file.
    x : str
        The name of the column in `df` to be used for the x-axis.
    y : str
        The name of the column in `df` to be used for the y-axis.
    style : str, optional
        The method for classifying the data. Options are 'quantile' or 'equal'. 
        Default is 'quantile', which divides the data into quantiles.
    n_bins : int, optional
        The number of bins to use for classification. Default is 3.
    cmap_name : str, optional
        The name of the colormap to use for coloring the points. 
        Default is 'viridis', but can be set to any valid matplotlib colormap name.
    figsize : tuple, optional
        A tuple specifying the size of the figure. Default is (8, 6).
    legend_kwargs : dict, optional
        A dictionary of keyword arguments passed to the legend creation function. 
        Can include 'legend_position' and 'legend_size'.
    plot_kwargs : dict, optional
        A dictionary of keyword arguments for the scatter plot, such as 'labelsize', 
        'ticklabelsize', 'title', and 'titlesize'.
    **plt_kwargs : 
        Additional keyword arguments passed to the matplotlib scatter function.
        
    Raises:
    ValueError: If the specified style is not 'quantile' or 'equal'.
    
    Example:
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = {'x': np.random.normal(50, 10, 2000), 'y': np.random.normal(50, 10, 2000)}
    >>> df = pd.DataFrame(data)
    >>> bivariate_plot(df, 'x', 'y', n_bins=4, cmap_name='bwr', alpha=0.7, edgecolor='black', 
    ... legend_kwargs={'legend_position': (1.10, 0.15), 'legend_size': 0.2})
    """
    
    labelsize = plot_kwargs.get('labelsize', 14)
    ticklabelsize = plot_kwargs.get('ticklabelsize', 12)
    title = plot_kwargs.get('title', f'Bivariate Plot: {x} vs {y}')
    titlesize = plot_kwargs.get('titlesize', 16)

    df = pd.read_csv(df_path).dropna()
    
    # Step 1: Classify the X variable
    if style == 'quantile':
        df[f'{x}_class'] = pd.qcut(df[x], q=n_bins, labels=False) + 1  # Start from 1 for labels
    elif style == 'equal':
        df[f'{x}_class'] = pd.cut(df[x], bins=n_bins, labels=False) + 1
    else:
        raise ValueError("Style must be 'quantile' or 'equal'.")

    # Step 2: Classify the Y variable
    if style == 'quantile':
        df[f'{y}_class'] = pd.qcut(df[y], q=n_bins, labels=False) + 1
    elif style == 'equal':
        df[f'{y}_class'] = pd.cut(df[y], bins=n_bins, labels=False) + 1

    # Step 3: Create a bivariate classification
    df['bivar_class'] = df[f'{x}_class'].astype(str) + "-" + df[f'{y}_class'].astype(str)

    # Step 4: Generate colors dynamically using the selected colormap
    cmap = plt.get_cmap(cmap_name)
    
    # Create a color list for unique bivariate classifications
    unique_classes = sorted(df['bivar_class'].unique())
    colors = [cmap(i / len(unique_classes)) for i in range(len(unique_classes))]  # Sample from cmap

    # Create a dictionary to map bivariate classifications to colors
    color_map = dict(zip(unique_classes, colors))  # Map classes to colors
    
    # Assign colors to the data based on the bivariate classification
    df['color'] = df['bivar_class'].map(color_map)

    # Get limits for x and y
    xlim = (df[x].min(), df[x].max())
    ylim = (df[y].min(), df[y].max())

    # Step 5: Plot the bivariate scatterplot using matplotlib
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df[x], df[y], color=df['color'], **plt_kwargs)
    
    # Add axis labels and title
    ax.set_xlabel(x, fontsize=labelsize)
    ax.set_ylabel(y, fontsize=labelsize)
    ax.set_title(title, fontsize=titlesize)
    
    # Set the ticks and labels for x and y axes based on limits
    ax.set_xticks(np.linspace(xlim[0], xlim[1], num=n_bins + 1))  # Create n_bins + 1 ticks
    ax.set_xticklabels([f'{tick:.1f}' for tick in np.linspace(xlim[0], xlim[1], num=n_bins + 1)], fontsize=ticklabelsize)  # Format x ticks

    ax.set_yticks(np.linspace(ylim[0], ylim[1], num=n_bins + 1))  # Create n_bins + 1 ticks
    ax.set_yticklabels([f'{tick:.1f}' for tick in np.linspace(ylim[0], ylim[1], num=n_bins + 1)], fontsize=ticklabelsize)  # Format y ticks

    # Create the custom square legend, passing the existing ax and legend_kwargs
    create_bivariate_legend(fig, ax, n_bins, colors, x_label=x, y_label=y, xlim=xlim, ylim=ylim, **(legend_kwargs or {}))

    plt.savefig(os.path.join(os.path.dirname(df_path), 'bivariate_df_plot.jpeg'), dpi=300, bbox_inches='tight')
    plt.show()

def create_bivariate_legend(fig, ax, n_bins, colors, x_label='X', y_label='Y', xlim=(0, 1), ylim=(0, 1), **legend_kwargs):
    """
    Create a custom square legend for the bivariate plot.
    
    Parameters:
    fig : matplotlib.figure.Figure
        The figure object to which the legend will be added.
    ax : matplotlib.axes.Axes
        The axes to draw the legend on.
    n_bins : int
        The number of bins for classification.
    colors : list
        The list of colors for the legend.
    x_label : str
        The label for the x-axis.
    y_label : str
        The label for the y-axis.
    """
    # Retrieve parameters from legend_kwargs with defaults
    legend_position = legend_kwargs.get('legend_position', (1.10, 0.15))
    legend_size = legend_kwargs.get('legend_size', 0.2)
    labelsize = legend_kwargs.get('labelsize', 12)
    ticklabelsize = legend_kwargs.get('ticklabelsize', 10)
    title = legend_kwargs.get('title', '')
    titlesize = legend_kwargs.get('titlesize', 14)

    # Ensure colors has the correct length for reshaping
    if len(colors) != (n_bins ** 2):
        raise ValueError(f"Expected {n_bins**2} colors, but got {len(colors)} colors.")

    # Reshape the colors into an n_bins x n_bins grid
    color_grid = np.array(colors)[:, :3].reshape(n_bins, n_bins, 3) 

    # Calculate the position of the legend based on the main axis
    bbox = ax.get_position()  # Get the position of the main axis
    legend_x = bbox.x0 + legend_position[0] * bbox.width  # x position relative to the main axis
    legend_y = bbox.y0 + legend_position[1] * bbox.height  # y position relative to the main axis

    # Create a rectangle for the legend
    legend_ax = fig.add_axes([legend_x, legend_y, legend_size, legend_size])  # Adjust the size as needed
    
    # Create the grid of squares
    for i in range(n_bins):
        for j in range(n_bins):
            legend_ax.add_patch(plt.Rectangle((i, j), 1, 1, color=color_grid[j, i], ec="gray"))

    # Set limits and remove ticks
    legend_ax.set_xlim(0, n_bins)
    legend_ax.set_ylim(0, n_bins)
    legend_ax.set_xticks(np.arange(n_bins) + 0.5)  # Set x-ticks at the center of each square
    legend_ax.set_xticklabels([f'{xlim[0] + (xlim[1] - xlim[0]) * (i + 0.5) / n_bins:.0f}' for i in range(n_bins)], fontsize=ticklabelsize)  # Format x ticks
    legend_ax.set_yticks(np.arange(n_bins) + 0.5)  # Set y-ticks at the center of each square
    legend_ax.set_yticklabels([f'{ylim[0] + (ylim[1] - ylim[0]) * (j + 0.5) / n_bins:.0f}' for j in range(n_bins)], fontsize=ticklabelsize)  # Format y ticks

    # Add labels for axes
    legend_ax.set_xlabel(x_label, ha='center', va='center', fontsize=labelsize, labelpad=10)
    legend_ax.set_ylabel(y_label, ha='center', va='center', fontsize=labelsize, labelpad=10)

    # Set title for the legend
    legend_ax.set_title(title, pad=20, fontsize=titlesize)
