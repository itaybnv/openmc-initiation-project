import numpy as np


def optimal_bins_fd(data):
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    bin_width = 2 * iqr / len(data) ** (1 / 3)
    n_bins = int(np.ceil((data.max() - data.min()) / bin_width))
    return max(n_bins, 1)  # At least 1 bin
