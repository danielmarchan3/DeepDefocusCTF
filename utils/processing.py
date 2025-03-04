import numpy as np
import os
import mrcfile
from concurrent.futures import ProcessPoolExecutor

import tensorflow as tf
from scipy.ndimage import rotate
from skimage.transform import rescale


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


def process_micrograph(micrograph_path, original_pixel_size, target_pixel_size=2, box_size=512):
    """
    Compute the averaged PSD from patches of a downsampled micrograph.

    Parameters:
    - micrograph_path (str): Path to the .mrc micrograph.
    - original_pixel_size (float): The original pixel size in Å.
    - target_pixel_size (float): The target pixel size in Å (default 2 Å).
    - box_size (int): Size of extracted patches.

    Returns:
    - avg_psd (np.array): The averaged Power Spectral Density (PSD).
    """
    try:
        with mrcfile.open(micrograph_path, permissive=True) as mrc:
            img = mrc.data.astype(np.float32)

        # Compute rescale factor
        rescale_factor = original_pixel_size / target_pixel_size

        # Downsample image
        downsampled_img = rescale(img, rescale_factor, anti_aliasing=True, preserve_range=True)

        # Extract patches from the downsampled image
        patches = extract_patches(downsampled_img, box_size=box_size)

        # Compute PSD for each patch and average them
        psds = np.array([compute_psd(patch) for patch in patches])
        avg_psd = np.mean(psds, axis=0)  # Average across patches

        return avg_psd

    except Exception as e:
        print(f"Error processing {micrograph_path}: {e}")
        return None


def process_micrographs_parallel(micrograph_paths, output_dir, original_pixel_size, target_pixel_size=2, box_size=512, num_workers=None):
    """
    Process multiple micrographs in parallel.

    Parameters:
    - micrograph_paths (list): List of paths to .mrc micrographs.
    - output_dir (path): Path where to store the psds
    - original_pixel_size (float): The original pixel size in Å.
    - target_pixel_size (float): The target pixel size in Å (default 2 Å).
    - box_size (int): Size of extracted patches.
    - num_workers (int): Number of parallel workers (default: use all available CPUs).
    """

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_micrograph, path, original_pixel_size, target_pixel_size, box_size): path for path in micrograph_paths}

        for future in futures:
            path = futures[future]
            try:
                result = future.result()
                if result is not None:
                    # Save the PSD as a .npy file
                    psd_filename = os.path.join(output_dir, os.path.basename(path).replace(".mrc", "_psd.npy"))
                    np.save(psd_filename, result)
            except Exception as e:
                print(f"Failed to process {path}: {e}")


def rotate_image(image_path, angle):
    """
    Rotate a .npy array and return both the rotated image and transformation matrix.

    Parameters:
    - image_path (str): Path to the .npy file.
    - angle (float): Rotation angle in degrees.

    Returns:
    - rotated_image (np.array): The rotated image.
    - transform_matrix (np.array): The transformation matrix.
    """
    # Load the .npy file
    image = np.load(image_path)

    # Rotate the image
    rotated_image = rotate(image, angle=angle, reshape=False)

    # Compute the transformation matrix for rotation
    theta = np.radians(angle)
    transform_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

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

def compute_ctf(kV, sampling_rate, size, defocusU, defocusV, Cs, phase_shift_PP, angle_ast):
    """Compute the Contrast Transfer Function (CTF) based on given parameters."""
    e_wavelength = 2.75e-2 if kV == 200 else 2.24e-2
    x, y = np.linspace(-1 / 2 * sampling_rate, 1 / 2 * sampling_rate, size), np.linspace(-1 / 2 * sampling_rate,
                                                                                         1 / 2 * sampling_rate, size)
    X, Y = np.meshgrid(x, y)

    return ctf_function(X, Y, e_wavelength, defocusU, defocusV, Cs, phase_shift_PP, angle_ast)


def ctf_function(x, y, e_wavelength, defocusU, defocusV, Cs, phase_shift_PP, angle_ast):
    """Compute the 2D CTF function."""
    angle_g = np.arctan2(y, x)
    dz = defocusU * (np.cos(angle_g - np.radians(angle_ast)) ** 2) + defocusV * (
                np.sin(angle_g - np.radians(angle_ast)) ** 2)
    freq = np.sqrt(x ** 2 + y ** 2)

    return ctf_1d(dz, lambda_e=e_wavelength, freq=freq, cs=Cs)


def ctf_1d(dz, lambda_e, freq, cs):
    """Compute the 1D CTF function."""
    term1 = np.pi * dz * lambda_e * (freq ** 2)
    term2 = np.pi / 2 * cs * (lambda_e ** 3) * (freq ** 4)

    return -np.cos(term1 - term2).astype(np.float32)


def compute_ctf_tf(kV, sampling_rate, size, defocusU, defocusV, Cs, phase_shift_PP, angle_ast):
    """Compute the Contrast Transfer Function (CTF) based on given parameters with TF library."""
    e_wavelength = 2.75e-2 if kV == 200 else 2.24e-2
    # Generate x, y values for a grid using TensorFlow
    x, y = (tf.linspace(-1 / 2 * sampling_rate, 1 / 2 * sampling_rate, size),
            tf.linspace(-1 / 2 * sampling_rate, 1 / 2 * sampling_rate, size))

    # Generate grid using TensorFlow meshgrid
    X, Y = tf.meshgrid(x, y)

    return  ctf_function_tf(X, Y, e_wavelength, defocusU, defocusV, Cs, phase_shift_PP, angle_ast)


def ctf_function_tf(x, y, e_wavelength, defocusU, defocusV, Cs, phase_shift_PP, angle_ast):
    angle_g = tf.atan2(y, x)
    # Assuming angle_ast is a tensor
    angle_ast = tf.multiply(angle_ast, np.pi / 180.0)
    dz = defocusU * tf.math.square(tf.math.cos(angle_g - angle_ast)) + defocusV * tf.math.square(tf.math.sin(angle_g - angle_ast))
    freq = tf.sqrt(tf.square(x) + tf.square(y))
    # print("TF - angle_g:", angle_g, "angle_ast:", angle_ast, "dz:", dz, "freq:", freq)

    return ctf_1d_tf(dz, lambda_e=e_wavelength, freq=freq, cs=Cs)


def ctf_1d_tf(dz, lambda_e, freq, cs):
    term1 = tf.multiply(tf.constant(np.pi, dtype=tf.float32), tf.multiply(tf.multiply(dz, lambda_e), tf.square(freq)))
    term2 = tf.multiply(tf.constant(np.pi / 2, dtype=tf.float32), tf.multiply(tf.multiply(cs, tf.pow(lambda_e, 3)), tf.pow(freq, 4)))
    # print("TF - term1:", term1, "term2:", term2)

    return -tf.cos(term1 - term2)

# ---------------------- MISCELLANEOUS UTILITIES ---------------------

def sum_angles(angle1, angle2):
    """Compute the sum of two angles, resetting to 0 if it reaches 180 degrees."""
    return (angle1 + angle2) % 180
