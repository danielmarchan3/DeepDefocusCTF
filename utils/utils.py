import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from skimage.transform import rescale
import mrcfile
from concurrent.futures import ProcessPoolExecutor


# ---------------------- TENSORFLOW UTILITIES --------------------------------------
def start_session():
    """Initialize TensorFlow session with dynamic memory allocation."""
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.keras.backend.clear_session()

    print('Enable dynamic memory allocation')
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)


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

                    #psd_results[os.path.basename(path)] = result
            except Exception as e:
                print(f"Failed to process {path}: {e}")

    #return psd_results


"""def rotate_image_old(image_path, angle):
    '''Rotate a np.array and return also the transformation matrix
    #imag: np.array
    #angle: angle in degrees
    #shape: output shape
    #P: transform matrix (further transformation in addition to the rotation)'''
    img = xmipp.Image(image_path)
    image = img.getData()

    rotated_image = rotate(image, angle=angle, reshape=False)

    image_transformed = xmipp.Image()
    image_transformed.setData(rotated_image)

    return image_transformed
    """

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

"""
def center_window_old(image_path, objective_res=2, sampling_rate=1):
    Extract a centered window from an image based on the given resolution and sampling rate.
    img = xmipp.Image(image_path)
    img_data = img.getData()
    x_dim = img_data.shape[1]

    window_size = int(x_dim * (sampling_rate / objective_res))
    half_window = window_size // 2
    center_x, center_y = img_data.shape[0] // 2, img_data.shape[1] // 2

    window_img = img.window2D(center_x - half_window + 1, center_y - half_window + 1,
                              center_x + half_window, center_y + half_window)

    window_data = window_img.getData()
    return (window_data - np.mean(window_data)) / np.std(window_data)
"""

# ---------------------- DATA GENERATION UTILITIES --------------------------------------

def prepare_test_data(df):
    """Prepare test data from a DataFrame."""
    Ndim = df.shape[0]
    img_matrix = np.zeros((Ndim, 512, 512, 1), dtype=np.float64)
    defocus_vector = np.zeros((Ndim, 2), dtype=np.float64)
    angle_vector = np.zeros((Ndim, 1), dtype=np.float64)

    for i, index in enumerate(df.index):
        #img_matrix[i, :, :, 0] = xmipp.Image(df.at[index, 'FILE']).getData()
        img_matrix[i, :, :, 0] = np.load(df.at[index, 'FILE'])
        defocus_vector[i] = [df.at[index, 'DEFOCUS_U'], df.at[index, 'DEFOCUS_V']]
        angle_vector[i, 0] = df.at[index, 'Angle']

    return img_matrix, defocus_vector, angle_vector


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


# ---------------------- PLOTTING UTILITIES --------------------------------------

def save_plot(plt_obj, folder, filename):
    """Save a plot to a file."""
    plt_obj.tight_layout()
    plt_obj.savefig(os.path.join(folder, filename))


def plot_histogram(data, title, folder, filename, bins=25, color='blue'):
    """Generate and save a histogram plot."""
    plt.figure()
    plt.hist(data, bins=bins, color=color, alpha=0.7, edgecolor='black')
    plt.title(title)
    plt.grid(True)
    save_plot(plt, folder, filename)


def plot_correlation(real, prediction, title, folder, filename):
    """Generate and save a scatter plot showing correlation between real and predicted values."""
    plt.figure(figsize=(10, 8))
    plt.scatter(real, prediction, alpha=0.7)
    plt.plot([0, max(real)], [0, max(real)], 'r--')  # Perfect correlation line
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(title)
    save_plot(plt, folder, filename)


def make_data_descriptive_plots(df_metadata, folder, COLUMNS, ground_truth=False):
    df_defocus = df_metadata[[COLUMNS['defocus_U'], COLUMNS['defocus_V']]]

    plot_histogram(df_defocus, 'Defocus histogram', folder, 'defocus_histogram.png')

    plot_correlation(df_metadata[COLUMNS['defocus_U']], df_metadata[COLUMNS['defocus_V']],
                     'Defocus correlation', folder ,'defocus_correlation.png')

    if ground_truth:
        df_defocus['ErrorU'] = df_metadata[COLUMNS['defocus_U']] - df_metadata['DEFOCUS_U_Est']
        df_defocus['ErrorV'] = df_metadata[COLUMNS['defocus_V']] - df_metadata['DEFOCUS_V_Est']

        df_defocus[['ErrorU', 'ErrorV']].plot.hist(alpha=0.5, bins=25)
        plt.title('Defocus error histogram')
        plt.savefig(os.path.join(folder, 'defocus_error_hist.png'))
        # BOXPLOT
        plt.figure()
        df_defocus[['ErrorU', 'ErrorV']].plot.box()
        plt.title('Defocus error boxplot')
        plt.savefig(os.path.join(folder, 'defocus_error_boxplot.png'))

    print(df_defocus.describe())

    df_angle = df_metadata[[COLUMNS['angle'], COLUMNS['cosAngle'], COLUMNS['sinAngle']]]
    plot_histogram(df_angle, 'Angle histogram', folder, 'angle_histogram.png')

    if ground_truth:
        df_angle_error = df_metadata[COLUMNS['angle']] - df_metadata['Angle_Est']
        plot_histogram(df_angle_error, 'Angle error histogram', folder, 'angle_error_histogram.png')
        print(df_angle_error.describe())


def make_training_plots(history, folder, prefix):
    # plot loss during training to CHECK OVERFITTING
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.plot(history.history['loss'], 'b', label='Training Loss')
    plt.plot(history.history['val_loss'], 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # Add grid lines for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    # Save the figure
    plt.savefig(os.path.join(folder, prefix + 'Training_and_Loss.png'))
    # plt.show()

    # Plot Learning Rate decreasing
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    # Plot Learning Rate
    plt.plot(history.epoch, history.history["lr"], "bo-", label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate", color='b')
    plt.tick_params(axis='y', colors='b')
    plt.grid(True)
    plt.title("Learning Rate and Validation Loss", fontsize=14)
    # Create a twin Axes sharing the xaxis
    ax2 = plt.gca().twinx()
    # Plot Validation Loss
    ax2.plot(history.epoch, history.history["val_loss"], "r^-", label="Validation Loss")
    ax2.set_ylabel('Validation Loss', color='r')
    ax2.tick_params(axis='y', colors='r')
    # Ensure both legends are displayed
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='upper left')
    # Save the figure
    plt.savefig(os.path.join(folder, prefix + 'Reduce_lr.png'))
    # plt.show()


def make_testing_plots(prediction, real, folder):
    # DEFOCUS PLOT
    plt.figure(figsize=(16, 8))  # Adjust the figure size as needed
    # Plot for Defocus U
    plt.subplot(211)
    plt.title('Defocus U')
    x = range(1, len(real[:, 0]) + 1)
    plt.scatter(x, real[:, 0], c='r', label='Real dU', marker='o')
    plt.scatter(x, prediction[:, 0], c='b', label='Predicted dU', marker='x')
    plt.xlabel("Sample Index")
    plt.ylabel("Defocus U")
    plt.grid(True)
    plt.legend()
    # Plot for Defocus V
    plt.subplot(212)
    plt.title('Defocus V')
    plt.scatter(x, real[:, 1], c='r', label='Real dV', marker='o')
    plt.scatter(x, prediction[:, 1], c='b', label='Predicted dV', marker='x')
    plt.xlabel("Sample Index")
    plt.ylabel("Defocus V")
    plt.grid(True)
    plt.legend()
    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(folder, 'predicted_vs_real_def.png'))

    # CORRELATION PLOT
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    # Plot for Defocus U
    plt.subplot(211)
    plt.title('Defocus U')
    plt.scatter(real[:, 0], prediction[:, 0])
    plt.plot([0, max(real[:, 0])], [0, max(real[:, 0])], color='red', linestyle='--')  # Line for perfect correlation
    plt.xlabel('True Values [defocus U]')
    plt.ylabel('Predictions [defocus U]')
    plt.xlim([min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])])
    plt.ylim([min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])])
    # Plot for Defocus V
    plt.subplot(212)
    plt.title('Defocus V')
    plt.scatter(real[:, 1], prediction[:, 1])
    plt.plot([0, max(real[:, 0])], [0, max(real[:, 0])], color='red', linestyle='--')  # Line for perfect correlation
    plt.xlabel('True Values [defocus V]')
    plt.ylabel('Predictions [defocus V]')
    plt.xlim([min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])])
    plt.ylim([min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])])
    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(folder, 'correlation_test_def.png'))
    # plt.show()

    # DEFOCUS ERROR
    # Plot for Defocus U
    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plt.title('Defocus U')
    error_u = prediction[:, 0] - real[:, 0]
    plt.hist(error_u, bins=25, color='blue', alpha=0.7)  # Adjust color and transparency
    plt.xlabel("Prediction Error Defocus U")
    plt.ylabel("Count")
    # Plot for Defocus V
    plt.subplot(212)
    plt.title('Defocus V')
    error_v = prediction[:, 1] - real[:, 1]
    plt.hist(error_v, bins=25, color='green', alpha=0.7)  # Adjust color and transparency
    plt.xlabel("Prediction Error Defocus V")
    plt.ylabel("Count")
    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(folder, 'defocus_prediction_error.png'))


def make_testing_angle_plots(prediction, real, folder):
    prediction = np.squeeze(prediction)
    real = np.squeeze(real)

    x = range(1, len(real) + 1)
    # DEFOCUS ANGLE PLOT
    plt.figure(figsize=(16, 8))
    # Plot for angle
    plt.title('Predicted vs real Angle)')
    plt.scatter(x, real, c='r', label='Real angle', marker='o')
    plt.scatter(x, prediction, c='b', label='Predicted angle', marker='x')
    plt.xlabel("Sample Index")
    plt.ylabel("Angle)")
    plt.legend()
    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(folder, 'predicted_vs_real_def_angle.png'))

    # DEFOCUS ANGLE PREDICTED VS REAL
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    # Plot for Angle
    plt.title('Correlation angle')
    plt.scatter(real, prediction)
    plt.xlabel('True Values angle')
    plt.ylabel('Predictions angle')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    plt.plot([0, max(real)], [0, max(real)], color='red', linestyle='--')  # Line for perfect correlation
    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(folder, 'correlation_test_def_angle.png'))

    # DEFOCUS ANGLE ERROR
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    # Plot for Angle
    plt.title('Angle Prediction Error')
    error_sin = prediction - real
    plt.hist(error_sin, bins=25, color='blue', alpha=0.7)  # Adjust color and transparency
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(folder, 'defocus_angle_prediction_error.png'))


def plot_training_history(history, folder, prefix):
    """Plot and save training and validation loss over epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], 'b', label='Training Loss')
    plt.plot(history.history['val_loss'], 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    save_plot(plt, folder, f'{prefix}_Training_and_Loss.png')


# ---------------------- MISCELLANEOUS UTILITIES --------------------------------------

def sum_angles(angle1, angle2):
    """Compute the sum of two angles, resetting to 0 if it reaches 180 degrees."""
    return (angle1 + angle2) % 180
