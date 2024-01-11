import matplotlib.pyplot as plt
import numpy as np
import missingno as msno
from matplotlib.colors import ListedColormap


def plot_msno(df):
    df = df.replace(0, np.nan)
    msno.matrix(df, labels=True, label_rotation=90)
    return


def plot_rank_matrix(rank_data):
    rank_pivot = rank_data.pivot_table(index='order_date', columns='market_maker_mnemonic', values='rank',
                                       aggfunc='mean')

    # re-arrange columns
    sorted_columns = rank_pivot.sum().sort_values(ascending=True).index
    rank_pivot = rank_pivot[sorted_columns]
    print(f'>>>> the company that has the highest non na columns:{list(rank_pivot.columns)[:10]}')

    def blend_with_white(color, weight=0.5):
        """Blend a color with white."""
        return tuple([(1 - weight) * channel + weight for channel in color])

    # Choose 10 distinct base colors from the 'tab10' colormap
    base_colors = plt.cm.tab10.colors

    # For each base color, generate 5 shades
    all_shades = []
    for base_color in base_colors:
        for weight in np.linspace(0, 0.8, 5):  # Blending factors to create 5 shades
            all_shades.append(blend_with_white(base_color, weight))

    # Create a colormap with the shades
    custom_cmap = ListedColormap(all_shades)

    # Plot the matrix
    fig, ax = plt.subplots(figsize=(15, 10))
    cax = ax.imshow(rank_pivot, cmap=custom_cmap, aspect='auto', interpolation='none')
    cbar = fig.colorbar(cax, ticks=np.arange(1, 51, 1), spacing='proportional')

    # Displaying the plot
    plt.title("Matrix Plot with Distinct Colors for Numbers 1-50")

    return plt


def validate_missing_bday(raw_order):
    # Missing BDay
    pass


def validate_duplicate_order_id(raw_order):
    pass


def validate_duplicate_rows(row_order):
    pass
    # Duplicate order ID
    # Duplicate rows