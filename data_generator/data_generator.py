import tensorflow as tf
import numpy as np
from utils.utils import center_window

class CustomDataGenPINN(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size, objective_res, sampling_rate):
        self.data = data
        self.batch_size = batch_size
        self.objective_res = objective_res
        self.sampling_rate = sampling_rate

    def on_epoch_end(self):
        self.data = self.data.sample(frac=1).reset_index(drop=True)

    def __get_input(self, path):
        imageMatrixNorm = center_window(path, objective_res=self.objective_res, sampling_rate=self.sampling_rate)
        return imageMatrixNorm

    def __get_output_defocus(self, defocus_scaled):
        return np.expand_dims(np.array(defocus_scaled), axis=-1)

    def __get_output_angle(self, angle):
        return np.expand_dims(np.array(angle), axis=-1)

    def __get_data(self, batches):
        image_batch = batches['FILE']
        defocus_U_batch_unscaled = batches['DEFOCUS_U_SCALED'].to_numpy()
        defocus_V_batch_unscaled = batches['DEFOCUS_V_SCALED'].to_numpy()
        defocus_angle_batch = batches['NORMALIZED_ANGLE'].to_numpy()

        X_batch = np.asarray([self.__get_input(x) for x in image_batch])
        # Ensure that all output arrays have the same number of dimensions
        yU_batch = np.asarray([self.__get_output_defocus(defocus_U) for defocus_U in defocus_U_batch_unscaled])
        yV_batch = np.asarray([self.__get_output_defocus(defocus_V) for defocus_V in defocus_V_batch_unscaled])
        yAngle_batch = np.asarray([self.__get_output_angle(angle) for angle in defocus_angle_batch])

        # Concatenate the four outputs
        y_batch = np.concatenate([yU_batch, yV_batch, yAngle_batch], axis=-1)

        return X_batch, y_batch

    def __getitem__(self, index):
        batches = self.data[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))