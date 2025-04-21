from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, \
    GlobalAveragePooling2D, Lambda, Concatenate, Reshape, UpSampling2D, MaxPool2D
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

import tensorflow as tf
from ..utils.metrics_and_model import angle_error_metric, custom_loss_CTF_with_scaler, mae_defocus_error, corr_CTF_metric

# ----------------------------- Models architecture -------------------------------------

class DeepDefocusMultiOutputModel():
    """
    Used to generate our multi-output model. This CNN contains two branches, one for defocus
    and another for the defocus angles. Each branch contains a sequence of Convolutional Layers that is defined
    on the make_default_hidden_layers method.
    """

    def __init__(self, width=512, height=512):
        self.IM_WIDTH = width
        self.IM_HEIGHT = height


    def build_defocus_branch(self, input, suffix):
        """
        Used to build the defocus in V branch of our multi-regression network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.        """

        h = Conv2D(filters=16, kernel_size=(8, 8), # 8, 8
                   activation='relu', padding='same', name='conv2d_1'+suffix)(input)
        h = Conv2D(filters=16, kernel_size=(8, 8),
                   activation='relu', padding='same', name='conv2d_2'+suffix)(h)
        h = BatchNormalization()(h)

        h = MaxPool2D(pool_size=(2, 2), name='pool_1'+suffix)(h)

        h = Conv2D(filters=16, kernel_size=(6, 6),
                   activation='relu', padding='same', name='conv2d_3'+suffix)(h)
        h = Conv2D(filters=16, kernel_size=(6, 6),
                   activation='relu', padding='same', name='conv2d_4'+suffix)(h)
        h = BatchNormalization()(h)

        h = MaxPool2D(pool_size=(3, 3), name='pool_2'+suffix)(h)

        h = Conv2D(filters=16, kernel_size=(2, 2),
                   activation='relu', padding='same', name='conv2d_5'+suffix)(h)
        h = Conv2D(filters=16, kernel_size=(2, 2),
                   activation='relu', padding='same', name='conv2d_6'+suffix)(h)
        h = BatchNormalization()(h)

        # h = Conv2D(filters=16, kernel_size=(2, 2),
        #            activation='relu', padding='same', name='conv2d_7' + suffix)(h)
        # h = Conv2D(filters=16, kernel_size=(2, 2),
        #            activation='relu', padding='same', name='conv2d_8' + suffix)(h)
        # h = BatchNormalization()(h)

        h = Flatten(name='flatten'+suffix)(h)

        return h

    def build_angle_branch(self, input, suffix):
        """
        Used to build the defocus in V branch of our multi-regression network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.        """

        h = Conv2D(filters=16, kernel_size=(64, 64),
                   activation='relu', padding='same', name='conv2d_1'+suffix)(input)
        #h = Conv2D(filters=16, kernel_size=(64, 64),
        #            activation='relu', padding='same', name='conv2d_2'+suffix)(h)
        h = BatchNormalization(epsilon=1e-5)(h)
        h = MaxPool2D(pool_size=(2, 2), name='pool_1'+suffix)(h)

        h = Conv2D(filters=16, kernel_size=(8, 8),
                   activation='relu', padding='same', name='conv2d_3'+suffix)(h)
        h = Conv2D(filters=16, kernel_size=(8, 8),
                    activation='relu', padding='same', name='conv2d_4'+suffix)(h)
        h = BatchNormalization(epsilon=1e-5)(h)

        h = MaxPool2D(pool_size=(3, 3), name='pool_2'+suffix)(h)

        h = Conv2D(filters=16, kernel_size=(4, 4),
                   activation='relu', padding='same', name='conv2d_5'+suffix)(h)
        h = BatchNormalization(epsilon=1e-5)(h)

        h = Flatten(name='flatten'+suffix)(h)

        return h


    def build_defocusU_branch(self, convLayer):
        L = Dense(32, activation='relu', kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l1_l2(0.001), name="denseU_1")(convLayer)
        L = Dropout(0.2, name="DropU_1")(L)
        L = Dense(16, activation='relu', name="denseU_2")(L)

        defocusU = Dense(1, activation='linear', name='defocus_U_output')(L)

        return defocusU


    def build_defocusV_branch(self, convLayer):
        L = Dense(32, activation='relu', kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l1_l2(0.001), name="denseV_1")(convLayer)
        L = Dropout(0.2, name="DropV_1")(L)
        L = Dense(16, activation='relu', name="denseV_2")(L)

        defocusV = Dense(1, activation='linear', name='defocus_V_output')(L)

        return defocusV


    def build_angle_dense_branch(self, convLayer):
        L = Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(0.01), name="denseAngle_1")(convLayer)
        L = Dropout(0.2, name="DropA_1")(L)
        L = Dense(16, activation='relu', name="denseAngle_2")(L)

        angle = Dense(1, activation='sigmoid', name='angle_output')(L)

        return angle

    def assemble_model_separated_defocus(self, width, height):
        input_shape = (height, width, 1)
        input_layer = Input(shape=input_shape, name='input')

        # DEFOCUS U and V
        defocus_branch_at_2 = self.build_defocus_branch(input_layer, suffix="defocus")
        L = Dropout(0.2)(defocus_branch_at_2)

        defocusU = self.build_defocusU_branch(L)
        defocusV = self.build_defocusV_branch(L)

        # DEFOCUS ANGLE
        defocus_angles_branch = self.build_angle_branch(input_layer, suffix="angle")
        La = Dropout(0.2)(defocus_angles_branch)

        angle = self.build_angle_dense_branch(La)

        # OUTPUT
        concatenated = Concatenate(name='ctf_values')([defocusU, defocusV, angle])

        model = Model(inputs=input_layer, outputs=concatenated,
                      name="deep_separated_defocus_net")

        return model

    # ----------- GET MODEL -------------------
    def getFullModel(self, learning_rate, defocus_scaler, cs, kV):
        model = self.assemble_model_separated_defocus(self.IM_WIDTH, self.IM_HEIGHT)
        model.summary()

        # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)  # Clip gradients
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=1.0)

        loss_custom = lambda y_true, y_pred: custom_loss_CTF_with_scaler(y_true, y_pred, defocus_scaler, cs, kV)

        def angle_error(y_true, y_pred):
            return angle_error_metric(y_true, y_pred)

        def mae_defocus(y_true, y_pred):
            return mae_defocus_error(y_true, y_pred, defocus_scaler)

        def corr_CTF(y_true, y_pred):
            return corr_CTF_metric(y_true, y_pred, defocus_scaler, cs, kV)

        #model.compile(optimizer=optimizer, loss=loss_custom, metrics=[angle_error, mae_defocus, corr_CTF], loss_weights=None)
        model.compile(optimizer=optimizer, loss='mae', metrics=[angle_error, mae_defocus, corr_CTF], loss_weights=None)

        return model

class DeepDefocusSimpleModel():
    """

    """

    def __init__(self, width=512, height=512):
        self.IM_WIDTH = width
        self.IM_HEIGHT = height

    def build_cnn_branch_chatgpt(self, input, suffix):
        h = Conv2D(filters=16, kernel_size=(11, 11),  # Larger kernel to capture broader Thon rings
                   activation='relu', padding='same', name='conv2d_1' + suffix)(input)
        h = BatchNormalization(epsilon=1e-5)(h)
        h = MaxPool2D(pool_size=(2, 2), name='pool_1' + suffix)(h)

        h = Conv2D(filters=32, kernel_size=(7, 7),
                   activation='relu', padding='same', name='conv2d_2' + suffix)(h)
        h = BatchNormalization(epsilon=1e-5)(h)
        h = MaxPool2D(pool_size=(3, 3), name='pool_2' + suffix)(h)

        h = Conv2D(filters=64, kernel_size=(3, 3),
                   activation='relu', padding='same', name='conv2d_3' + suffix)(h)
        h = BatchNormalization(epsilon=1e-5)(h)

        h = Flatten(name='flatten' + suffix)(h)

        return h

    def build_cnn_branch(self, input, suffix):
        """
        Used to build the defocus in V branch of our multi-regression network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.        """

        h = Conv2D(filters=16, kernel_size=(8, 8),  # 8, 8
                   activation='relu', padding='same', name='conv2d_1' + suffix)(input)
        h = Conv2D(filters=16, kernel_size=(8, 8),
                   activation='relu', padding='same', name='conv2d_2'+suffix)(h)
        h = BatchNormalization()(h)

        h = MaxPool2D(pool_size=(2, 2), name='pool_1' + suffix)(h)

        h = Conv2D(filters=16, kernel_size=(6, 6),
                   activation='relu', padding='same', name='conv2d_3' + suffix)(h)
        h = Conv2D(filters=16, kernel_size=(6, 6),
                   activation='relu', padding='same', name='conv2d_4'+suffix)(h)
        h = BatchNormalization()(h)

        h = MaxPool2D(pool_size=(3, 3), name='pool_2' + suffix)(h)

        h = Conv2D(filters=16, kernel_size=(2, 2),
                   activation='relu', padding='same', name='conv2d_5' + suffix)(h)
        h = Conv2D(filters=16, kernel_size=(2, 2),
                   activation='relu', padding='same', name='conv2d_6'+suffix)(h)
        h = BatchNormalization()(h)

        # h = Conv2D(filters=16, kernel_size=(2, 2),
        #            activation='relu', padding='same', name='conv2d_7' + suffix)(h)
        # h = Conv2D(filters=16, kernel_size=(2, 2),
        #            activation='relu', padding='same', name='conv2d_8' + suffix)(h)
        # h = BatchNormalization()(h)

        h = Flatten(name='flatten' + suffix)(h)

        return h

    def build_defocusU_branch(self, convLayer):
        L = Dense(32, activation='relu', kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l1_l2(0.001), name="denseU_1")(convLayer)
        L = Dropout(0.2, name="DropU_1")(L)
        L = Dense(16, activation='relu', name="denseU_2")(L)

        defocusU = Dense(1, activation='linear', name='defocus_U_output')(L)

        return defocusU


    def build_defocusV_branch(self, convLayer):
        L = Dense(32, activation='relu', kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l1_l2(0.001), name="denseV_1")(convLayer)
        L = Dropout(0.2, name="DropV_1")(L)
        L = Dense(16, activation='relu', name="denseV_2")(L)

        defocusV = Dense(1, activation='linear', name='defocus_V_output')(L)

        return defocusV

    def build_angle_branch(self, convLayer):
        L = Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(0.01), name="denseAngle_1")(convLayer)
        L = Dropout(0.2, name="DropA_1")(L)
        L = Dense(16, activation='relu', name="denseAngle_2")(L)

        angle = Dense(1, activation='sigmoid', name='angle_output')(L)

        return angle

    def assemble_model_separated_defocus(self, width, height):
        input_shape = (height, width, 1)
        input_layer = Input(shape=input_shape, name='input')

        # DEFOCUS U and V
        cnn_branch = self.build_cnn_branch(input_layer, suffix="cnn")
        L = Dropout(0.2)(cnn_branch)

        defocusU = self.build_defocusU_branch(L)
        defocusV = self.build_defocusV_branch(L)
        angle = self.build_angle_branch(L)

        # OUTPUT
        concatenated = Concatenate(name='ctf_values')([defocusU, defocusV, angle])

        model = Model(inputs=input_layer, outputs=concatenated,
                      name="deep_defocus_net")

        return model

    # ----------- GET MODEL -------------------
    def getFullModel(self, learning_rate, defocus_scaler, cs, kV):
        model = self.assemble_model_separated_defocus(self.IM_WIDTH, self.IM_HEIGHT)
        model.summary()

        # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)  # Clip gradients
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=1.0)

        loss_custom = lambda y_true, y_pred: custom_loss_CTF_with_scaler(y_true, y_pred, defocus_scaler, cs, kV)

        def angle_error(y_true, y_pred):
            return angle_error_metric(y_true, y_pred)

        def mae_defocus(y_true, y_pred):
            return mae_defocus_error(y_true, y_pred, defocus_scaler)

        def corr_CTF(y_true, y_pred):
            return corr_CTF_metric(y_true, y_pred, defocus_scaler, cs, kV)

        #model.compile(optimizer=optimizer, loss=loss_custom, metrics=[mae_defocus, angle_error, corr_CTF], loss_weights=None)
        model.compile(optimizer=optimizer, loss='mae', metrics=[mae_defocus, angle_error, corr_CTF], loss_weights=None)

        return model

# --------------------------- To extract models captured features ------------------------------------------------
from utils.processing import center_window
import os
import matplotlib.pyplot as plt

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