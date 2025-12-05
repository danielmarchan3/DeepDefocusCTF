import numpy as np
import os
import mrcfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import tensorflow as tf
from scipy.ndimage import rotate
from skimage.transform import rescale
from sklearn.preprocessing import StandardScaler


# ----------------------- DATA PROCESSING UTILITIES --------------------------------------

def fit_log_scaler_from_columns(*columns):
    """
    Fit a StandardScaler on the log10 of multiple defocus columns.

    Parameters:
        *columns: Multiple 1D arrays or Series of defocus values

    Returns:
        scaler: Fitted StandardScaler object
    """
    # Stack all columns vertically
    log_values = np.log10(np.vstack([np.asarray(col).reshape(-1, 1) for col in columns]))

    scaler = StandardScaler()
    scaler.fit(log_values)
    return scaler


def transform_with_log_scaler(column, scaler):
    """
    Transform a single defocus column using a previously fitted log-scaler.

    Parameters:
        column: 1D array or Series (defocus)
        scaler: StandardScaler fitted on log10 values

    Returns:
        scaled_column: The transformed values
    """
    log_vals = np.log10(np.asarray(column).reshape(-1, 1))
    return scaler.transform(log_vals)


# ---------------------- IMAGE PROCESSING UTILITIES --------------------------------------

def extract_patches(image, box_size):
    """Extract non-overlapping patches of size (box_size, box_size) from the image."""
    patches = []
    img_height, img_width = image.shape
    for i in range(0, img_height - box_size + 1, box_size):
        for j in range(0, img_width - box_size + 1, box_size):
            patches.append(image[i:i + box_size, j:j + box_size])
    return patches


def compute_psd(patch):
    """Compute the Power Spectral Density (PSD) of a given patch."""
    fft_result = np.fft.fft2(patch)
    psd = np.abs(fft_result) ** 2
    psd = np.fft.fftshift(psd)  # Shift the zero frequency component to the center
    return np.log1p(psd)  # Apply log scaling for better visualization


def process_micrograph(micrograph_path, output_path, original_pixel_size, target_pixel_size=1, box_size=512):
    """
    Process a single micrograph and save the averaged Power Spectral Density (PSD) to the specified output path (.npy).

    Parameters:
    - micrograph_path (str): Path to the input .mrc micrograph file.
    - output_path (str): Path where the computed PSD will be saved as a .npy file.
    - original_pixel_size (float): Original pixel size of the micrograph in Ångströms.
    - target_pixel_size (float, default=1): Target pixel size for downsampling before PSD computation.
    - box_size (int, default=512): Size of patches extracted from the downsampled micrograph for PSD calculation.

    Behavior:
    - Loads the micrograph from the .mrc file and ensures it is a single 2D image.
    - Downsamples the image to match the target pixel size using anti-aliasing.
    - Extracts patches of size `box_size` from the downsampled image.
    - Computes the PSD for each patch and averages them to obtain the final PSD.
    - Saves the averaged PSD as a .npy file at `output_path`.
    - Returns True if successful, False otherwise.
    """
    
    try:
        with mrcfile.open(micrograph_path, permissive=True) as mrc:
            img = mrc.data.astype(np.float32)

        # Check dimensions
        if img.ndim == 3:
            if img.shape[0] == 1:
                img = img[0]
            else:
                raise ValueError(f"MRC contains a stack with {img.shape[0]} slices; expected a single 2D micrograph.")
        elif img.ndim != 2:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        # Rescale
        rescale_factor = original_pixel_size / target_pixel_size
        downsampled_img = rescale(img, rescale_factor, anti_aliasing=True, preserve_range=True)

        # Extract patches and promediate PSD
        patches = extract_patches(downsampled_img, box_size=box_size)
        psds = np.array([compute_psd(patch) for patch in patches])
        avg_psd = np.mean(psds, axis=0)

        # Save in output_path
        np.save(output_path, avg_psd)
        return True

    except Exception as e:
        print(f"Error processing {micrograph_path}: {e}")
        return False


def process_micrographs_parallel(mic_pairs, original_pixel_size, target_pixel_size=1, box_size=512, num_workers=None):

    """
    Process multiple micrographs in parallel and save PSDs in output_dir.

    Parameters:
    - micrograph_paths (list): List of paths to .mrc micrograph files.
    - output_dir (str): Directory where PSD .npy files will be saved.
    - original_pixel_size (float): Original pixel size of the micrographs in Ångströms.
    - target_pixel_size (float, default=1): Target pixel size for downsampling before PSD computation.
    - box_size (int, default=512): Size of patches extracted from each micrograph for PSD calculation.
    - num_workers (int or None): Number of parallel workers (defaults to all available CPUs if None).
    - file_to_id (dict or None): Optional mapping {micrograph_path: ID} to generate unique PSD filenames
      using the pattern ID_basename_psd.npy. If not provided, filenames will use only the basename.

    Behavior:
    - Uses ProcessPoolExecutor to compute PSDs in parallel by calling process_micrograph for each micrograph.
    - For each result, saves the PSD as a .npy file in output_dir:
      - If file_to_id is provided: ID_basename_psd.npy.
      - Otherwise: basename_psd.npy.
    """

    if num_workers is None:
        num_workers = os.cpu_count() or 1

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_micrograph, in_path, out_path, original_pixel_size, target_pixel_size, box_size): (in_path, out_path)
            for (in_path, out_path) in mic_pairs
        }

        for future in as_completed(futures):
            in_path, out_path = futures[future]
            try:
                success = future.result()
                if not success:
                    print(f"Failed to process {in_path} -> {out_path}")
            except Exception as e:
                print(f"Exception processing {in_path} -> {out_path}: {e}")



def rotate_image(image_path, angle):
    """
    Rotate a .npy array and return both the rotated image and transformation matrix.

    Parameters:
    - image_path (str): Path to the .npy file.
    - angle (float): Rotation angle in degrees.

    Returns:
    - rotated_image (np.array): The rotated image.
    """
    # Load the .npy file
    image = np.load(image_path)
    rotated_image = rotate(image, angle=angle, reshape=False)

    return rotated_image


def center_window(image_path, objective_res=2, sampling_rate=1):
    """
    Extract a centered window from a .npy image based on the given resolution and sampling rate.

    Parameters:
    - image_path (str): Path to the .npy file.
    - objective_res (float): Target resolution.
    - sampling_rate (float): Pixel sampling rate.

    Returns:
    - window_data (np.array): The extracted and normalized window.
    """
    # Load the .npy file
    img_data = np.load(image_path)

    # Image dimensions
    x_dim, y_dim = img_data.shape  # Assuming 2D image

    # Compute window size
    window_size = int(x_dim * (sampling_rate / objective_res))
    half_window = window_size // 2

    # Center coordinates
    center_x, center_y = x_dim // 2, y_dim // 2

    # Extract window
    window_data = img_data[
                  center_x - half_window: center_x + half_window,
                  center_y - half_window: center_y + half_window
                  ]

    # Normalize window (zero mean, unit variance)
    window_data = (window_data - np.mean(window_data)) / np.std(window_data)

    return window_data


# ---------------------- CTF COMPUTATION UTILITIES --------------------------------------
def compute_ctf_tf(kV, sampling_rate, size, defocusU, defocusV, Cs, phase_shift_PP, angle_ast):
    """Compute the Contrast Transfer Function (CTF) based on given parameters with TensorFlow."""
    # Electron wavelength in nm (200kV or 300kV)
    e_wavelength = tf.constant(2.75e-2 if kV == 200 else 2.24e-2, dtype=tf.float32)
    # Generate x, y values for a grid using TensorFlow
    x = tf.linspace(-0.5 / sampling_rate, 0.5 / sampling_rate, size)
    y = tf.linspace(-0.5 / sampling_rate, 0.5 / sampling_rate, size)
    # Generate grid using TensorFlow meshgrid
    X, Y = tf.meshgrid(x, y)

    return ctf_function_tf(X, Y, e_wavelength, defocusU, defocusV, Cs, phase_shift_PP, angle_ast)


def ctf_function_tf(x, y, e_wavelength, defocusU, defocusV, Cs, phase_shift_PP, angle_ast):
    """Compute the CTF function in 2D."""

    # Ensure all inputs are TensorFlow float32
    x, y, e_wavelength, defocusU, defocusV, Cs, phase_shift_PP = map(
        lambda v: tf.cast(v, tf.float32), [x, y, e_wavelength, defocusU, defocusV, Cs, phase_shift_PP])

    angle_g = tf.atan2(y, x)  # Compute angles in Fourier space
    # Convert `angle_ast` to radians and cast to TensorFlow float32
    angle_ast = tf.cast(angle_ast, tf.float32) * tf.constant(np.pi / 180.0, dtype=tf.float32)
    # Compute defocus variation in 2D
    dz = defocusU * tf.square(tf.cos(angle_g - angle_ast)) + defocusV * tf.square(tf.sin(angle_g - angle_ast))
    # Compute frequency radius
    freq = tf.sqrt(tf.square(x) + tf.square(y))
    # print("TF - angle_g:", angle_g, "angle_ast:", angle_ast, "dz:", dz, "freq:", freq)

    return ctf_1d_tf(dz, lambda_e=e_wavelength, freq=freq, cs=Cs)


def ctf_1d_tf(dz, lambda_e, freq, cs):
    """Compute the 1D CTF equation for given defocus and frequency values."""

    # Ensure float32 consistency
    dz, lambda_e, freq, cs = map(tf.cast, [dz, lambda_e, freq, cs], [tf.float32] * 4)

    term1 = np.pi * dz * lambda_e * tf.square(freq)  # Defocus term
    term2 = (np.pi / 2) * cs * tf.pow(lambda_e, 3) * tf.pow(freq, 4)  # Spherical aberration term
    # print("TF - term1:", term1, "term2:", term2)

    return -tf.cos(term1 - term2)  # Final CTF value


# ---------------------- MISCELLANEOUS UTILITIES ---------------------

def sum_angles(angle1, angle2):
    """Compute the sum of two angles, resetting to 0 if it reaches 180 degrees."""
    return (angle1 + angle2) % 180


def pearson_correlation_ts(array1, array2, epsilon):
    """Compute Pearson correlation between two TensorFlow tensors."""
    ctf_array_true_flat = tf.reshape(array1, [-1])
    ctf_array_pred_flat = tf.reshape(array2, [-1])

    # Mean-centered vectors
    mean_true = tf.reduce_mean(ctf_array_true_flat)
    mean_pred = tf.reduce_mean(ctf_array_pred_flat)

    centered_true = ctf_array_true_flat - mean_true
    centered_pred = ctf_array_pred_flat - mean_pred

    # Pearson correlation formula
    numerator = tf.reduce_sum(tf.multiply(centered_true, centered_pred))
    denominator_true = tf.sqrt(tf.reduce_sum(tf.square(centered_true)))
    denominator_pred = tf.sqrt(tf.reduce_sum(tf.square(centered_pred)))

    return numerator / (denominator_true * denominator_pred + epsilon)
