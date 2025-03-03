from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.utils import center_window, compute_ctf, compute_ctf_tf
import os
import numpy as np
from scipy.stats import pearsonr
from deep_defocus_model import DeepDefocusMultiOutputModel

# ------------------------- Metrics to evaluate the Models loss during training ----------------------------------------

def angle_error_metric(y_true, y_pred):
    # Extract angles predicted and true values from the model's output
    angle_true = y_true[:, 2] * 180
    angle_pred = y_pred[:, 2] * 180
    # Calculate the absolute error in degrees
    angle_error_degrees = K.abs(angle_pred - angle_true)
    # Return the mean angle error
    return K.mean(angle_error_degrees)

def mae_defocus_error(y_true, y_pred, defocus_scaler):
    median_ = defocus_scaler.center_
    iqr_ = defocus_scaler.scale_

    y_true_unscaled = (y_true[:, 0:2] * iqr_) + median_
    y_pred_unscaled = (y_pred[:, 0:2] * iqr_) + median_

    metric_value = tf.reduce_mean(tf.abs(y_true_unscaled - y_pred_unscaled))

    return metric_value

def corr_CTF_metric(y_true, y_pred, defocus_scaler, cs, kV):
    sampling_rate = 1 # TODO Risky to put it here
    size = 512
    epsilon = 1e-8

    # Extract unscaled defocus values from y_true
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    angle_true = y_true[:, 2]
    angle_pred = y_pred[:, 2]

    median_ = defocus_scaler.center_
    iqr_ = defocus_scaler.scale_

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

        # Flatten the arrays to make them 1D
        # ctf_array_true_flat = tf.reshape(ctf_array_true, [-1])
        # ctf_array_pred_flat = tf.reshape(ctf_array_pred, [-1])
        #
        # # Calculate mean-centered vectors
        # mean_true = tf.reduce_mean(ctf_array_true_flat)
        # mean_pred = tf.reduce_mean(ctf_array_pred_flat)
        #
        # centered_true = ctf_array_true_flat - mean_true
        # centered_pred = ctf_array_pred_flat - mean_pred
        #
        # # Calculate Pearson correlation coefficient
        # numerator = tf.reduce_sum(tf.multiply(centered_true, centered_pred))
        # denominator_true = tf.sqrt(tf.reduce_sum(tf.square(centered_true)))
        # denominator_pred = tf.sqrt(tf.reduce_sum(tf.square(centered_pred)))
        #
        # correlation_coefficient = numerator / (denominator_true * denominator_pred + epsilon)
        # correlation_coefficient_loss = 1 - correlation_coefficient

        # return correlation_coefficient_loss
        return tf.abs(ctf_array_true - ctf_array_pred)

    elementwise_losses = tf.map_fn(lambda x: elementwise_loss(x[0], x[1], x[2], x[3], x[4], x[5]),
                                   (defocus_U_true, defocus_U_pred, defocus_V_true, defocus_V_pred,
                                    angle_true, angle_pred),
                                   dtype=tf.float32)

    return tf.reduce_mean(elementwise_losses)

# ------------------------- Custom loss function for CTF based on a mathematical formula -------------------------------

def custom_loss_CTF_with_scaler(y_true, y_pred, defocus_scaler, cs, kV):
    sampling_rate = 1
    size = 512
    epsilon = 1e-8
    #print("Shape of y_true:", y_true.shape)
    #print("Shape of y_pred:", y_pred.shape)

    # Extract unscaled defocus values from y_true
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    defocus_U_true_scaled = y_true[:, 0]  # Assuming defocus_U is the first output
    defocus_V_true_scaled = y_true[:, 1]  # Assuming defocus_V is the second output
    angle_true = y_true[:, 2]

    defocus_U_pred_scaled = y_pred[:, 0]  # Assuming defocus_U is the first output
    defocus_V_pred_scaled = y_pred[:, 1]  # Assuming defocus_V is the second output
    angle_pred = y_pred[:, 2]

    median_ = defocus_scaler.center_
    iqr_ = defocus_scaler.scale_

    y_true_unscaled = (y_true * iqr_) + median_
    y_pred_unscaled = (y_pred * iqr_) + median_

    #y_true_unscaled = y_true * defocus_scaler
    #y_pred_unscaled = y_pred * defocus_scaler

    # Example: Access individual output tensors
    defocus_U_true = y_true_unscaled[:, 0]  # Assuming defocus_U is the first output
    defocus_V_true = y_true_unscaled[:, 1]  # Assuming defocus_V is the second output
    defocus_U_pred = y_pred_unscaled[:, 0]  # Assuming defocus_U is the first output
    defocus_V_pred = y_pred_unscaled[:, 1]  # Assuming defocus_V is the second output

    # ------
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

        # # Flatten the arrays to make them 1D
        # ctf_array_true_flat = tf.reshape(ctf_array_true, [-1])
        # ctf_array_pred_flat = tf.reshape(ctf_array_pred, [-1])
        #
        # # Calculate mean-centered vectors
        # mean_true = tf.reduce_mean(ctf_array_true_flat)
        # mean_pred = tf.reduce_mean(ctf_array_pred_flat)
        #
        # centered_true = ctf_array_true_flat - mean_true
        # centered_pred = ctf_array_pred_flat - mean_pred
        #
        # # Calculate Pearson correlation coefficient
        # numerator = tf.reduce_sum(tf.multiply(centered_true, centered_pred))
        # denominator_true = tf.sqrt(tf.reduce_sum(tf.square(centered_true)))
        # denominator_pred = tf.sqrt(tf.reduce_sum(tf.square(centered_pred)))
        #
        # correlation_coefficient = numerator / (denominator_true * denominator_pred + epsilon)
        # correlation_coefficient_loss = 1 - correlation_coefficient

        #return correlation_coefficient_loss
        return tf.abs(ctf_array_true - ctf_array_pred) # MSE or MAE

    #tf.print("defocus_U_true:", defocus_U_true)
    #tf.print("defocus_V_true:", defocus_V_true)
    #tf.print("defocus_U_pred:", defocus_U_pred)
    #tf.print("defocus_V_pred:", defocus_V_pred)
    #tf.print("Angle_true:", angle_true)
    #tf.print("Angle_pred:", angle_pred)

    elementwise_losses = tf.map_fn(lambda x: elementwise_loss(x[0], x[1], x[2], x[3], x[4], x[5]),
                                   (defocus_U_true, defocus_U_pred, defocus_V_true, defocus_V_pred,
                                    angle_true, angle_pred),
                                   dtype=tf.float32)

    defocus_U_loss = tf.reduce_mean(tf.abs(defocus_U_true_scaled - defocus_U_pred_scaled))
    defocus_V_loss = tf.reduce_mean(tf.abs(defocus_V_true_scaled - defocus_V_pred_scaled))
    defocus_loss = tf.reduce_mean([defocus_U_loss, defocus_V_loss])
    angle_loss = tf.reduce_mean(tf.abs(angle_true - angle_pred))
    image_loss = tf.reduce_mean(elementwise_losses)

    weight_image_loss = 0.5 # 0.8
    weight_defocus_loss = 1
    weight_angle_loss = 0.7 # 0.8

    # Aggregate the elementwise losses
    aggregated_loss = image_loss * weight_image_loss + defocus_loss * weight_defocus_loss + angle_loss * weight_angle_loss

    return aggregated_loss

# -------------------------------- Utils to test the ctf function approach --------------------------------------------

def exampleCTFApplyingFunction(df_metadata):
    defocusU = df_metadata.head(1)['DEFOCUS_U'].values[0]
    defocusV = df_metadata.head(1)['DEFOCUS_V'].values[0]
    kV = df_metadata.head(1)['kV'].values[0]
    defocusA = df_metadata.head(1)['Angle'].values[0]

    cs = 2.7e7
    sampling_rate = 1
    size = 512
    epsilon = 1e-8

    ctf_array_ts = compute_ctf_tf(kV=kV, sampling_rate=sampling_rate, size=size, defocusU=defocusU, defocusV=defocusV, Cs=cs,
                                        phase_shift_PP=0, angle_ast=defocusA)
    print(defocusU, defocusV, defocusA, cs, sampling_rate, kV)

    ctf_array2_ts = compute_ctf_tf(kV=kV, sampling_rate=sampling_rate, size=size, defocusU=defocusU, defocusV=defocusV, Cs=cs,
                                         phase_shift_PP=0, angle_ast=defocusA + 45)

    ctf_array = compute_ctf(kV=kV, sampling_rate=sampling_rate, size=size, defocusU=defocusU,
                                   defocusV=defocusV, Cs=cs,
                                   phase_shift_PP=0, angle_ast=defocusA)

    ctf_array2 = compute_ctf(kV=kV, sampling_rate=sampling_rate, size=size, defocusU=defocusU,
                                   defocusV=defocusV, Cs=cs,
                                   phase_shift_PP=0, angle_ast=defocusA + 45)

    def pearson_correlation_ts(array1, array2):
        # Flatten the arrays to make them 1D
        ctf_array_true_flat = tf.reshape(array1, [-1])
        ctf_array_pred_flat = tf.reshape(array2, [-1])

        # Calculate mean-centered vectors
        mean_true = tf.reduce_mean(ctf_array_true_flat)
        mean_pred = tf.reduce_mean(ctf_array_pred_flat)

        centered_true = ctf_array_true_flat - mean_true
        centered_pred = ctf_array_pred_flat - mean_pred

        # Calculate Pearson correlation coefficient
        numerator = tf.reduce_sum(tf.multiply(centered_true, centered_pred))
        denominator_true = tf.sqrt(tf.reduce_sum(tf.square(centered_true)))
        denominator_pred = tf.sqrt(tf.reduce_sum(tf.square(centered_pred)))

        correlation_coefficient = numerator / (denominator_true * denominator_pred + epsilon)

        return correlation_coefficient

    def pearson_correlation(array1, array2):
        # Flatten the images into 1D arrays
        flat_image1 = array1.flatten()
        flat_image2 = array2.flatten()

        # Calculate Pearson correlation coefficient
        correlation_coefficient, p_value = pearsonr(flat_image1, flat_image2)

        return correlation_coefficient

    correlation_coefficient_TS = pearson_correlation_ts(ctf_array_ts, ctf_array2_ts)
    print(correlation_coefficient_TS)

    correlation_coefficient = pearson_correlation(ctf_array, ctf_array2)
    print(correlation_coefficient)

    # Plot the first image
    plt.figure()
    # plt.subplot(1, 2, 1)
    plt.imshow(ctf_array, cmap='gray')
    plt.title('Image 1')

    # Overlay the second image on top of the first
    #plt.subplot(1, 2, 2)
    #plt.imshow(ctf_array_ts.numpy() - ctf_array2_ts.numpy(), cmap='gray')  # Background image
    # plt.imshow(ctf_array2, cmap='viridis', alpha=0.5)  # Overlay image with some transparency
    #plt.title('Difference')
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


# --------------------------- To extract models captured features ------------------------------------------------

def extract_CNN_layer_features(modelDir, image_example_path, layers, defocus_scaler, objective_res, sampling_rate, xDim):
    one_image_data = center_window(image_example_path, objective_res=objective_res, sampling_rate=sampling_rate)
    one_image_data = one_image_data.reshape((-1, xDim, xDim, 1))

    model_defocus = DeepDefocusMultiOutputModel(width=xDim, height=xDim).getFullModel(learning_rate=0.001, defocus_scaler=defocus_scaler, cs=2.7e7, kV=200)
    model_defocus.load_weights(filepath=os.path.join(modelDir, 'Best_Weights'))
    features_path = os.path.join(modelDir, "featuresExtraction/")

    try:
        os.makedirs(features_path)
    except FileExistsError:
        pass

    model_defocus_layers = model_defocus.layers
    model_defocus_input = model_defocus.input

    layer_outputs_defocus = [layer.output for layer in model_defocus_layers]
    features_defocus_model = Model(inputs=model_defocus_input, outputs=layer_outputs_defocus)

    extracted_benchmark = features_defocus_model(one_image_data)

    # For the input image
    f1_benchmark = extracted_benchmark[0]
    print('\n Input benchmark shape:', f1_benchmark.shape)
    imgs = f1_benchmark[0, ...]
    plt.figure(figsize=(5, 5))
    plt.imshow(imgs[..., 0], cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.savefig(os.path.join(features_path, "features_layer_%s" % str(0)))
    # For the rest of the layers
    for layer in range(1, layers, 1):
        feature_benchmark = extracted_benchmark[layer]
        print('\n feature_benchmark shape:', feature_benchmark.shape)
        print('Layer ', layer)
        filters = feature_benchmark.shape[-1]
        imgs = feature_benchmark[0, ...]

        # Dynamically adjust the number of rows and columns based on the number of filters
        rows = 2
        cols = int(filters / 2) if filters % 2 == 0 else int(filters / 2) + 1
        # Dynamically adjust figsize based on the number of columns
        figsize = (cols * 5, rows * 5)

        plt.figure(figsize=figsize)
        for n in range(filters):
            ax = plt.subplot(rows, cols, n + 1)
            plt.imshow(imgs[..., n], cmap='gray')
            plt.axis('off')

        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        plt.savefig(os.path.join(features_path, "features_layer_%s" % str(layer)))
        plt.close()

    # Get intermediate activations
    # activations = features_defocus_model.predict(one_image_data)

    # # Visualize feature maps for some intermediate layers
    # for i, activation in enumerate(extracted_benchmark):
    #     if len(activation.shape) == 4:  # Check if the activation is from a convolutional layer
    #         plt.figure()
    #         for j in range(activation.shape[3]):  # Iterate over channels
    #             plt.subplot(4, 8, j + 1)
    #             plt.imshow(activation[0, :, :, j], cmap='gray')
    #             plt.axis('off')
    #         plt.suptitle(f'Layer {i}', fontsize=16)
    #         plt.show()