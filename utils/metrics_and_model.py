import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

from utils.processing import compute_ctf_tf, pearson_correlation_ts


# ---------------------- TENSORFLOW UTILITIES --------------------------------------

def start_session():
    """Initialize TensorFlow session with dynamic memory allocation."""
    """Clears previous TensorFlow session to free resources."""
    tf.keras.backend.clear_session()
    print("Cleared previous TensorFlow session.")
    tf.keras.backend.set_floatx('float32')

# ---------------------- DATA GENERATION UTILITIES --------------------------------------

def prepare_test_data(df):
    """Prepare test data from a DataFrame."""
    Ndim = df.shape[0]
    img_matrix = np.zeros((Ndim, 512, 512, 1), dtype=np.float64)
    defocus_vector = np.zeros((Ndim, 2), dtype=np.float64)
    angle_vector = np.zeros((Ndim, 1), dtype=np.float64)

    for i, index in enumerate(df.index):
        img_matrix[i, :, :, 0] = np.load(df.at[index, 'FILE'])
        defocus_vector[i] = [df.at[index, 'DEFOCUS_U'], df.at[index, 'DEFOCUS_V']]
        angle_vector[i, 0] = df.at[index, 'Angle']

    return img_matrix, defocus_vector, angle_vector


# ------------------------- Metrics to evaluate the Models loss during training ----------------------------------------

def angle_error_metric(y_true, y_pred):
    # Ensure float32 casting
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # Extract angles and convert to degrees
    angle_true = y_true[:, 2] * tf.constant(180.0, dtype=tf.float32)
    angle_pred = y_pred[:, 2] * tf.constant(180.0, dtype=tf.float32)
    # Calculate absolute error
    angle_error_degrees = K.abs(angle_pred - angle_true)
    # Return mean error
    return K.mean(angle_error_degrees)


def mae_defocus_error(y_true, y_pred, defocus_scaler):
    # Ensure y_true and y_pred are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # Convert defocus_scaler parameters to tensors with float32 dtype
    median_ = tf.convert_to_tensor(defocus_scaler.center_, dtype=tf.float32)
    iqr_ = tf.convert_to_tensor(np.clip(defocus_scaler.scale_, 1e-5, None), dtype=tf.float32)
    # Apply unscaling transformation
    y_true_unscaled = (y_true[:, 0:2] * iqr_) + median_
    y_pred_unscaled = (y_pred[:, 0:2] * iqr_) + median_
    # Compute mean absolute error
    metric_value = tf.reduce_mean(tf.abs(y_true_unscaled - y_pred_unscaled))

    return metric_value


def corr_CTF_metric(y_true, y_pred, defocus_scaler, cs, kV):
    sampling_rate = 2 # TODO Risky to put it here
    size = 512
    epsilon = 1e-8

    # Extract unscaled defocus values from y_true
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    angle_true = y_true[:, 2]
    angle_pred = y_pred[:, 2]

    # Ensure scaler is float32
    median_ = tf.convert_to_tensor(defocus_scaler.center_, dtype=tf.float32)
    iqr_ = tf.convert_to_tensor(np.clip(defocus_scaler.scale_, 1e-5, None), dtype=tf.float32)
    y_true_unscaled = (y_true * iqr_) + median_
    y_pred_unscaled = (y_pred * iqr_) + median_

    defocus_U_true = y_true_unscaled[:, 0]
    defocus_V_true = y_true_unscaled[:, 1]
    defocus_U_pred = y_pred_unscaled[:, 0]
    defocus_V_pred = y_pred_unscaled[:, 1]

    def elementwise_loss(defocus_U_true, defocus_U_pred, defocus_V_true, defocus_V_pred,
                         angle_true, angle_pred):
        # Extract true sin and cos values
        # Calculate the true angle
        true_angle = angle_true * 180
        pred_angle = angle_pred * 180

        ctf_array_true = compute_ctf_tf(kV=kV, sampling_rate=sampling_rate, size=size, defocusU=defocus_U_true,
                                              defocusV=defocus_V_true, Cs=cs, phase_shift_PP=0, angle_ast=true_angle)

        ctf_array_pred = compute_ctf_tf(kV=kV, sampling_rate=sampling_rate, size=size, defocusU=defocus_U_pred,
                                              defocusV=defocus_V_pred, Cs=cs, phase_shift_PP=0, angle_ast=pred_angle)

        correlation_coefficient = pearson_correlation_ts(ctf_array_true, ctf_array_pred, epsilon)
        correlation_coefficient_loss = 1 - correlation_coefficient

        return correlation_coefficient_loss
        # return tf.abs(ctf_array_true - ctf_array_pred)

    elementwise_losses = tf.map_fn(lambda x: elementwise_loss(x[0], x[1], x[2], x[3], x[4], x[5]),
                                   (defocus_U_true, defocus_U_pred, defocus_V_true, defocus_V_pred,
                                    angle_true, angle_pred),
                                   dtype=tf.float32)

    return tf.reduce_mean(elementwise_losses)

# ------------------------- Custom loss function for CTF based on a mathematical formula -------------------------------

def custom_loss_CTF_with_scaler(y_true, y_pred, defocus_scaler, cs, kV):
    sampling_rate = 2
    size = 512
    epsilon = 1e-8

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    defocus_U_true_scaled = y_true[:, 0]
    defocus_V_true_scaled = y_true[:, 1]
    angle_true = y_true[:, 2]

    defocus_U_pred_scaled = y_pred[:, 0]
    defocus_V_pred_scaled = y_pred[:, 1]
    angle_pred = y_pred[:, 2]

    median_ = tf.convert_to_tensor(defocus_scaler.center_, dtype=tf.float32)
    iqr_ = tf.convert_to_tensor(defocus_scaler.scale_, dtype=tf.float32)

    y_true_unscaled = (y_true * iqr_) + median_
    y_pred_unscaled = (y_pred * iqr_) + median_

    defocus_U_true = y_true_unscaled[:, 0]
    defocus_V_true = y_true_unscaled[:, 1]
    defocus_U_pred = y_pred_unscaled[:, 0]
    defocus_V_pred = y_pred_unscaled[:, 1]

    def elementwise_loss(defocus_U_true, defocus_U_pred, defocus_V_true, defocus_V_pred,
                         angle_true, angle_pred):
        true_angle = angle_true * 180
        pred_angle = angle_pred * 180

        ctf_array_true = compute_ctf_tf(kV=kV, sampling_rate=sampling_rate, size=size,
                                        defocusU=defocus_U_true, defocusV=defocus_V_true,
                                        Cs=cs, phase_shift_PP=0, angle_ast=true_angle)

        ctf_array_pred = compute_ctf_tf(kV=kV, sampling_rate=sampling_rate, size=size,
                                        defocusU=defocus_U_pred, defocusV=defocus_V_pred,
                                        Cs=cs, phase_shift_PP=0, angle_ast=pred_angle)

        correlation_coefficient = pearson_correlation_ts(ctf_array_true, ctf_array_pred, epsilon)
        correlation_coefficient_loss = 1 - correlation_coefficient

        return correlation_coefficient_loss
        # return tf.abs(ctf_array_true - ctf_array_pred)

    elementwise_losses = tf.map_fn(lambda x: elementwise_loss(x[0], x[1], x[2], x[3], x[4], x[5]),
                                   (defocus_U_true, defocus_U_pred, defocus_V_true, defocus_V_pred,
                                    angle_true, angle_pred),
                                   dtype=tf.float32)

    # Debugging alternative: Instead of tf.print, use assertions
    tf.debugging.assert_all_finite(defocus_U_true, "defocus_U_true has NaNs or Infs")
    tf.debugging.assert_all_finite(defocus_U_pred, "defocus_U_pred has NaNs or Infs")
    tf.debugging.assert_all_finite(defocus_V_true, "defocus_V_true has NaNs or Infs")
    tf.debugging.assert_all_finite(defocus_V_pred, "defocus_V_pred has NaNs or Infs")
    tf.debugging.assert_all_finite(angle_true, "angle_true has NaNs or Infs")
    tf.debugging.assert_all_finite(angle_pred, "angle_pred has NaNs or Infs")

    defocus_U_loss = tf.reduce_mean(tf.abs(defocus_U_true_scaled - defocus_U_pred_scaled))
    defocus_V_loss = tf.reduce_mean(tf.abs(defocus_V_true_scaled - defocus_V_pred_scaled))
    defocus_loss = tf.reduce_mean([defocus_U_loss, defocus_V_loss])
    angle_loss = tf.reduce_mean(tf.abs(angle_true - angle_pred)) # 0 - 1
    image_loss = tf.reduce_mean(elementwise_losses) # -1 - 1

    weight_image_loss = 0.5
    weight_defocus_loss = 2
    weight_angle_loss = 1

    aggregated_loss = defocus_loss * weight_defocus_loss + angle_loss * weight_angle_loss + image_loss * weight_image_loss

    return aggregated_loss

def custom_loss_CTF_with_scaler_old(y_true, y_pred, defocus_scaler, cs, kV):
    sampling_rate = 2
    size = 512
    epsilon = 1e-8

    # Extract unscaled defocus values from y_true
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    defocus_U_true_scaled = y_true[:, 0]  # Assuming defocus_U is the first output
    defocus_V_true_scaled = y_true[:, 1]  # Assuming defocus_V is the second output
    angle_true = y_true[:, 2]

    defocus_U_pred_scaled = y_pred[:, 0]  # Assuming defocus_U is the first output
    defocus_V_pred_scaled = y_pred[:, 1]  # Assuming defocus_V is the second output
    angle_pred = y_pred[:, 2]

    # Ensure scaler is float32
    median_ = tf.convert_to_tensor(defocus_scaler.center_, dtype=tf.float32)
    iqr_ = tf.convert_to_tensor(defocus_scaler.scale_, dtype=tf.float32)
    # iqr_ = tf.convert_to_tensor(np.clip(defocus_scaler.scale_, 1e-5, None), dtype=tf.float32)
    y_true_unscaled = (y_true * iqr_) + median_
    y_pred_unscaled = (y_pred * iqr_) + median_

    #y_true_unscaled = y_true * defocus_scaler
    #y_pred_unscaled = y_pred * defocus_scaler

    # Access individual output tensors
    defocus_U_true = y_true_unscaled[:, 0]  # Assuming defocus_U is the first output
    defocus_V_true = y_true_unscaled[:, 1]  # Assuming defocus_V is the second output
    defocus_U_pred = y_pred_unscaled[:, 0]  # Assuming defocus_U is the first output
    defocus_V_pred = y_pred_unscaled[:, 1]  # Assuming defocus_V is the second output

    def elementwise_loss(defocus_U_true, defocus_U_pred, defocus_V_true, defocus_V_pred,
                         angle_true, angle_pred):
        # Extract true sin and cos values
        # Calculate the true angle
        true_angle = angle_true * 180
        pred_angle = angle_pred * 180

        ctf_array_true = compute_ctf_tf(kV=kV, sampling_rate=sampling_rate, size=size, defocusU=defocus_U_true,
                                              defocusV=defocus_V_true, Cs=cs, phase_shift_PP=0, angle_ast=true_angle)

        ctf_array_pred = compute_ctf_tf(kV=kV, sampling_rate=sampling_rate, size=size, defocusU=defocus_U_pred,
                                              defocusV=defocus_V_pred, Cs=cs, phase_shift_PP=0, angle_ast=pred_angle)

        # Print intermediate values for debugging
        #tf.print("ctf_array_true:", ctf_array_true)
        #tf.print("ctf_array_pred:", ctf_array_pred)
        correlation_coefficient = pearson_correlation_ts(ctf_array_true, ctf_array_pred, epsilon)
        correlation_coefficient_loss = 1 - correlation_coefficient
        tf.print("ctf_correlation:", correlation_coefficient)

        #return correlation_coefficient_loss
        return tf.abs(ctf_array_true - ctf_array_pred) # MSE or MAE

    tf.print("defocus_U_true:", defocus_U_true)
    tf.print("defocus_V_true:", defocus_V_true)
    tf.print("defocus_U_pred:", defocus_U_pred)
    tf.print("defocus_V_pred:", defocus_V_pred)
    tf.print("Angle_true:", angle_true)
    tf.print("Angle_pred:", angle_pred)

    elementwise_losses = tf.map_fn(lambda x: elementwise_loss(x[0], x[1], x[2], x[3], x[4], x[5]),
                                   (defocus_U_true, defocus_U_pred, defocus_V_true, defocus_V_pred,
                                    angle_true, angle_pred),
                                   dtype=tf.float32)

    defocus_U_loss = tf.reduce_mean(tf.abs(defocus_U_true_scaled - defocus_U_pred_scaled))
    defocus_V_loss = tf.reduce_mean(tf.abs(defocus_V_true_scaled - defocus_V_pred_scaled))
    defocus_loss = tf.reduce_mean([defocus_U_loss, defocus_V_loss])
    angle_loss = tf.reduce_mean(tf.abs(angle_true - angle_pred))
    image_loss = tf.reduce_mean(elementwise_losses)

    weight_image_loss = 0.8
    weight_defocus_loss = 2
    weight_angle_loss = 0.5

    # Aggregate the elementwise losses
    # aggregated_loss = image_loss * weight_image_loss + defocus_loss * weight_defocus_loss + angle_loss * weight_angle_loss
    aggregated_loss = defocus_loss * weight_defocus_loss + angle_loss * weight_angle_loss

    return aggregated_loss

# -------------------------------- Utils to test the ctf function approach --------------------------------------------

def exampleCTFApplyingFunction(df_metadata):
    """Function to test TensorFlow CTF implementations."""

    # Extracting values from DataFrame and ensuring they are float
    psd_fn = df_metadata.head(1)['FILE'].values[0]
    defocusU = float(df_metadata.head(1)['DEFOCUS_U'].values[0])
    defocusV = float(df_metadata.head(1)['DEFOCUS_V'].values[0])
    kV = float(df_metadata.head(1)['kV'].values[0])
    defocusA = float(df_metadata.head(1)['Angle'].values[0])

    cs = 2.7e7
    sampling_rate = 1
    size = 512
    epsilon = 1e-8  # To prevent division by zero in Pearson correlation

    print(defocusU, defocusV, defocusA, cs, sampling_rate, kV)

    # Compute CTF using TensorFlow functions
    ctf_array_ts = compute_ctf_tf(kV=kV, sampling_rate=sampling_rate, size=size,
                                  defocusU=defocusU, defocusV=defocusV, Cs=cs,
                                  phase_shift_PP=0, angle_ast=defocusA)

    #ctf_array2_ts = compute_ctf_tf(kV=kV, sampling_rate=sampling_rate, size=size,
    #                               defocusU=defocusU, defocusV=defocusV, Cs=cs,
    #                               phase_shift_PP=0, angle_ast=defocusA + 45)

    ctf_array2_ts = np.load(psd_fn)

    # Compute correlation
    correlation_coefficient_TS = pearson_correlation_ts(ctf_array_ts, ctf_array2_ts, epsilon)
    print("TensorFlow Pearson Correlation:",
          correlation_coefficient_TS.numpy())  # Convert TF tensor to NumPy for printing

    # Plot numpy image
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(ctf_array_ts.numpy(), cmap='gray')
    plt.title('Image TS')

    # Plot ts image
    plt.subplot(1, 2, 2)
    plt.imshow(ctf_array2_ts, cmap='gray')
    plt.title('Image 2 TS')
    plt.show()


# -------------------------------------- Test this is for the training loss ---------------------------------------------

class CosineAnnealingScheduler(Callback):
    def __init__(self, initial_learning_rate, max_epochs, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.max_epochs = max_epochs
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'learning_rate'):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
        lr = self.initial_learning_rate * 0.5 * (1 + tf.math.cos(np.pi * epoch / self.max_epochs))
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        if self.verbose > 0:
            print(f'\nEpoch {epoch+1}/{self.max_epochs}, Learning Rate: {lr:.6f}')


