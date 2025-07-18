import numpy as np
import rasterio
from glob import glob


def calculate_feature_stats(image_directory):
    """
    Calculate the mean and standard deviation of a feature.

    Args:
        image_directory (str): Directory where the source GeoTIFF files are stored that are passed to model for training.
        image_pattern (str): Pattern of the GeoTIFF file names that globs files for computing stats.
        bands (list, optional): List of bands to calculate statistics for. Defaults to [0, 1, 2, 3, 4, 5].

    Raises:
        Exception: If no images are found in the given directory.

    Returns:
        tuple: Two lists containing the means and standard deviations of each band.
    """
    # Initialize lists to store the means and standard deviations
    all_means = []
    all_stds = []

    # Use glob to get a list of all .tif images in the directory
    all_images = glob(f"{image_directory}/{image_pattern}")

    # Make sure there are images to process
    if not all_images:
        raise Exception("No images found")

    # Get the number of bands
    num_bands = len(bands)

    # Initialize arrays to hold sums and sum of squares for each band
    band_sums = np.zeros(num_bands)
    band_sq_sums = np.zeros(num_bands)
    pixel_counts = np.zeros(num_bands)

    # Iterate over each image
    for image_file in all_images:
        with rasterio.open(image_file) as src:
            # For each band, calculate the sum, square sum, and pixel count
            for band in bands:
                data = src.read(band + 1)  # rasterio band index starts from 1
                band_sums[band] += np.nansum(data)
                band_sq_sums[band] += np.nansum(data**2)
                pixel_counts[band] += np.count_nonzero(~np.isnan(data))

    # Calculate means and standard deviations for each band
    for i in bands:
        mean = band_sums[i] / pixel_counts[i]
        std = np.sqrt((band_sq_sums[i] / pixel_counts[i]) - (mean**2))
        all_means.append(mean)
        all_stds.append(std)

    return all_means, all_stds