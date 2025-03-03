from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, \
    GlobalAveragePooling2D, Lambda, Concatenate, Reshape, UpSampling2D, MaxPool2D
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
import tensorflow as tf
from metrics_and_utils import *

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

    def build_defocus_branch_new(self, input, suffix):
        """
        Used to build the defocus in V branch of our multi-regression network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.        """

        h = Conv2D(filters=16, kernel_size=(8, 8), # 8, 8
                   activation='relu', padding='same', name='conv2d_1'+suffix)(input)
        #h = Conv2D(filters=16, kernel_size=(8, 8),
        #           activation='relu', padding='same', name='conv2d_2'+suffix)(h)
        h = BatchNormalization()(h)

        h = MaxPool2D(pool_size=(2, 2), name='pool_1'+suffix)(h)

        h = Conv2D(filters=16, kernel_size=(6, 6),
                   activation='relu', padding='same', name='conv2d_3'+suffix)(h)
        #h = Conv2D(filters=16, kernel_size=(6, 6),
        #           activation='relu', padding='same', name='conv2d_4'+suffix)(h)
        h = BatchNormalization()(h)

        h = MaxPool2D(pool_size=(3, 3), name='pool_2'+suffix)(h)

        h = Conv2D(filters=16, kernel_size=(2, 2),
                   activation='relu', padding='same', name='conv2d_5'+suffix)(h)
        #h = Conv2D(filters=16, kernel_size=(2, 2),
        #           activation='relu', padding='same', name='conv2d_6'+suffix)(h)
        h = BatchNormalization()(h)

        # h = Conv2D(filters=16, kernel_size=(2, 2),
        #            activation='relu', padding='same', name='conv2d_7' + suffix)(h)
        # h = Conv2D(filters=16, kernel_size=(2, 2),
        #            activation='relu', padding='same', name='conv2d_8' + suffix)(h)
        # h = BatchNormalization()(h)

        h = Flatten(name='flatten'+suffix)(h)

        return h

    def build_angle_branch_new(self, input, suffix):
        """
        Used to build the defocus in V branch of our multi-regression network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks,
        followed by the Dense output layer.        """

        h = Conv2D(filters=16, kernel_size=(64, 64),
                   activation='relu', padding='same', name='conv2d_1'+suffix)(input)
        h = BatchNormalization()(h)

        h = MaxPool2D(pool_size=(2, 2), name='pool_1'+suffix)(h)

        h = Conv2D(filters=16, kernel_size=(8, 8),
                   activation='relu', padding='same', name='conv2d_3'+suffix)(h)
        # h = Conv2D(filters=16, kernel_size=(8, 8),
        #            activation='relu', padding='same', name='conv2d_4'+suffix)(h)
        h = BatchNormalization()(h)

        h = MaxPool2D(pool_size=(3, 3), name='pool_2'+suffix)(h)

        h = Conv2D(filters=16, kernel_size=(4, 4),
                   activation='relu', padding='same', name='conv2d_5'+suffix)(h)
        h = BatchNormalization()(h)

        h = Flatten(name='flatten'+suffix)(h)

        return h

    def build_defocusU_branch(self, convLayer):
        L = Dense(32, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l1_l2(0.01),
                  name="denseU_1")(convLayer)
        L = Dropout(0.1, name="DropU_1")(L)
        L = Dense(16, activation='relu', name="denseU_2")(L)

        defocusU = Dense(1, activation='linear', name='defocus_U_output')(L)

        return defocusU

    def build_defocusV_branch(self, convLayer):
        L = Dense(32, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l1_l2(0.01),
                  name="denseV_1")(convLayer)
        L = Dropout(0.1, name="DropV_1")(L)
        L = Dense(16, activation='relu', name="denseV_2")(L)

        defocusV = Dense(1, activation='linear', name='defocus_V_output')(L)

        return defocusV

    def build_angle_branch(self, convLayer):
        L = Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(0.01), name="denseAngle_1")(convLayer)
        L = Dropout(0.1, name="DropA_1")(L)
        L = Dense(16, activation='relu', name="denseAngle_2")(L)

        angle = Dense(1, activation='sigmoid', name='angle_output')(L)

        return angle

    def assemble_model_separated_defocus(self, width, height):
        input_shape = (height, width, 1)
        input_layer = Input(shape=input_shape, name='input')

        # DEFOCUS U and V
        defocus_branch_at_2 = self.build_defocus_branch_new(input_layer, suffix="defocus")
        L = Dropout(0.2)(defocus_branch_at_2)

        defocusU = self.build_defocusU_branch(L)
        defocusV = self.build_defocusV_branch(L)

        # DEFOCUS ANGLE
        defocus_angles_branch = self.build_angle_branch_new(input_layer, suffix="angle")
        La = Dropout(0.2)(defocus_angles_branch)

        angle = self.build_angle_branch(La)

        # OUTPUT
        concatenated = Concatenate(name='ctf_values')([defocusU, defocusV, angle])

        model = Model(inputs=input_layer, outputs=concatenated,
                      name="deep_separated_defocus_net")

        return model

    # ----------- GET MODEL -------------------
    def getFullModel(self, learning_rate, defocus_scaler, cs, kV):
        model = self.assemble_model_separated_defocus(self.IM_WIDTH, self.IM_HEIGHT)
        model.summary()

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=1.0)

        loss = lambda y_true, y_pred: custom_loss_CTF_with_scaler(y_true, y_pred, defocus_scaler, cs, kV)

        def angle_error(y_true, y_pred):
            return angle_error_metric(y_true, y_pred)

        def mae_defocus(y_true, y_pred):
            return mae_defocus_error(y_true, y_pred, defocus_scaler)

        def corr_CTF(y_true, y_pred):
            return corr_CTF_metric(y_true, y_pred, defocus_scaler, cs, kV)

        model.compile(optimizer=optimizer, loss=loss, metrics=[angle_error, mae_defocus, corr_CTF], loss_weights=None)

        return model