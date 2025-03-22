import numpy as np
import tensorflow as tf

from utils.processing import center_window
import threading

lock = threading.Lock()  # Prevent concurrency issues when loading images

class CustomDataGenPINN(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size, objective_res, sampling_rate):
        self.data = data.copy()  # Prevent modification during training
        self.batch_size = batch_size
        self.objective_res = objective_res
        self.sampling_rate = sampling_rate

    def on_epoch_end(self):
        """Shuffle data at the end of each epoch to improve generalization."""
        self.data = self.data.sample(frac=1).reset_index(drop=True)

    def __get_input(self, path):
        """Loads and normalizes the image data."""
        with lock:  # Prevent multiple threads from loading the same file simultaneously
            try:
                imageMatrixNorm = center_window(path, objective_res=self.objective_res, sampling_rate=self.sampling_rate)
                if imageMatrixNorm is None or np.isnan(imageMatrixNorm).any():
                    raise ValueError(f"Invalid image at {path}")
            except Exception as e:
                raise RuntimeError(f"Error loading image {path}: {e}")

        # Ensure image shape is (H, W, 1)
        if len(imageMatrixNorm.shape) == 2:  # If grayscale
            imageMatrixNorm = np.expand_dims(imageMatrixNorm, axis=-1)
        return imageMatrixNorm.astype(np.float32)

    def __get_output_defocus(self, defocus_scaled):
        """Reshapes defocus values for training."""
        return np.array(defocus_scaled, dtype=np.float32).reshape(-1, 1)

    def __get_output_angle(self, angle):
        """Reshapes angle values for training."""
        return np.array(angle, dtype=np.float32).reshape(-1, 1)

    def __get_data(self, batches):
        """Retrieves a batch of images and labels."""
        image_batch = batches['FILE']
        defocus_U_batch_unscaled = batches['DEFOCUS_U_SCALED'].to_numpy()
        defocus_V_batch_unscaled = batches['DEFOCUS_V_SCALED'].to_numpy()
        defocus_angle_batch = batches['NORMALIZED_ANGLE'].to_numpy()

        X_batch = np.asarray([self.__get_input(x) for x in image_batch], dtype=np.float32)
        yU_batch = defocus_U_batch_unscaled.reshape(-1, 1)
        yV_batch = defocus_V_batch_unscaled.reshape(-1, 1)
        yAngle_batch = defocus_angle_batch.reshape(-1, 1)

        y_batch = np.concatenate([yU_batch, yV_batch, yAngle_batch], axis=-1)

        # Ensure no NaNs in batch
        if np.isnan(X_batch).any() or np.isnan(y_batch).any():
            raise ValueError("NaNs detected in batch.")

        return X_batch, y_batch

    def __getitem__(self, index):
        """Fetches a batch by index."""
        batches = self.data.iloc[index * self.batch_size:(index + 1) * self.batch_size]

        if batches.empty:
            raise ValueError(f"Error: Empty batch at index {index}. Check dataset size.")

        X, y = self.__get_data(batches)

        return X, y

    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(len(self.data) / self.batch_size))


class CustomDataGenPINN_old(tf.keras.utils.Sequence):
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