import os
import sys
import numpy as np
from time import time, sleep
import datetime
import pandas as pd

import tensorflow as tf
import tensorflow.keras.callbacks as callbacks
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from utils.metrics_and_model import (start_session, prepare_test_data, exampleCTFApplyingFunction, CosineAnnealingScheduler)
from utils.plotting import (make_data_descriptive_plots, make_training_plots, make_testing_plots, make_testing_angle_plots)
from utils.processing import (fit_log_scaler_from_columns, transform_with_log_scaler)
from data_generator.data_generator import CustomDataGenPINN, CustomDataGenPINN_old
from models.deep_defocus_model import DeepDefocusMultiOutputModel, DeepDefocusSimpleModel

BATCH_SIZE = 32
EPOCHS = 300
TEST_SIZE = 0.10
LEARNING_RATE_DEF = 0.0001

COLUMNS = {'id': 'ID', 'defocus_U': 'DEFOCUS_U', 'defocus_V': 'DEFOCUS_V',
           'sinAngle': 'Sin(2*angle)', 'cosAngle': 'Cos(2*angle)',
           'angle': 'Angle', 'resolution':'Resolution','kV': 'kV', 'file': 'FILE'}


# ------------------------ MAIN PROGRAM -----------------------------
if __name__ == "__main__":

    # ------------------------------------- PARSING DATA ----------------------------------------------------------------
    if len(sys.argv) < 3:
        print("Usage: python3 train_model.py <metadataDir> <modelDir>")
        sys.exit()

    metadataDir = sys.argv[1]
    modelDir = sys.argv[2]
    input_size = (256, 256, 1) # We have 512x512 PSD images at 1A resolution, so we are staying only with the center region of 2A
    objective_res = 2
    sampling_rate = 1 # The PSDs of our images have a sampling rate of 1A
    ground_Truth = False
    testing_Bool = True
    plots_Bool = True

    # -------------------------------------- INITIALIZING SYSTEM ------------------------------------------------------
    # Step 1: GPU Configuration (MUST be done before TensorFlow starts)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)  # Enable memory growth
            print("GPU memory growth enabled.")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")

    # Step 2: Start session (clear any old session)
    start_session()

    # ---------------------------------------- LOADING DATA -----------------------------------------------------------
    print("Loading data...")
    path_metadata = os.path.join(metadataDir, "metadata.csv")
    df_metadata = pd.read_csv(path_metadata)

    os.makedirs(modelDir, exist_ok=True)

    # Fit scaler on all defocus input values
    scaler = fit_log_scaler_from_columns(
        df_metadata[COLUMNS['defocus_U']],
        df_metadata[COLUMNS['defocus_V']]
    )

    # Apply scaler to transform each input defocus value
    df_metadata['DEFOCUS_U_SCALED'] = transform_with_log_scaler(df_metadata[COLUMNS['defocus_U']], scaler)
    df_metadata['DEFOCUS_V_SCALED'] = transform_with_log_scaler(df_metadata[COLUMNS['defocus_V']], scaler)

    df_metadata['NORMALIZED_ANGLE'] = df_metadata[COLUMNS['angle']]/180

    # ----------------------------------------------- STATISTICS -------------------------------------------------------

    print(df_metadata.describe())
    print(df_metadata.isnull().any())

    # -------------------------------------------- DESCRIPTIVE PLOTS ---------------------------------------------------

    if plots_Bool:
        make_data_descriptive_plots(df_metadata, modelDir, COLUMNS, True, ground_Truth)

    # -------------------------------------- SPLIT DATA: TRAIN, VALIDATE and TEST --------------------------------------

    df_training, df_test = train_test_split(df_metadata, test_size=TEST_SIZE)
    df_train, df_validate = train_test_split(df_training, test_size=0.20)

    # ------------------------------------- TEST CTF FUNCTION IMPLEMENTATION -------------------------------------------

    #exampleCTFApplyingFunction(df_train)

    # ------------------------------------------- TRAINING MODELS ------------------------------------------------------

    # OJO: The number of batches is equal to len(df)//batch_size
    traingen = CustomDataGenPINN(data=df_train.head(1984), batch_size=BATCH_SIZE,
                                objective_res=objective_res, sampling_rate=sampling_rate)

    valgen = CustomDataGenPINN(data=df_validate.head(480), batch_size=BATCH_SIZE,
                                objective_res=objective_res, sampling_rate=sampling_rate)

    testgen = CustomDataGenPINN(data=df_test, batch_size=1,
                                objective_res=objective_res, sampling_rate=sampling_rate)

    path_logs_defocus = os.path.join(modelDir, "logs_defocus/" + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

    callbacks_list_def = [
        callbacks.CSVLogger(os.path.join(path_logs_defocus, 'defocus.csv'), separator=',', append=False),
        callbacks.TensorBoard(log_dir=path_logs_defocus, histogram_freq=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1,
                                    mode='auto',
                                    min_delta=0.0001, cooldown=2, min_lr=0.000001),
        # CosineAnnealingScheduler(LEARNING_RATE_DEF, EPOCHS, verbose=1),
        callbacks.EarlyStopping(monitor='val_loss', patience=20),
        callbacks.ModelCheckpoint(filepath=os.path.join(path_logs_defocus, 'Best_Weights.weights.h5'),
                                      save_weights_only=True,
                                      save_best_only=True,
                                      monitor='val_loss',
                                      verbose=0)
        ]


    # Check if GPUs are available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Create a MirroredStrategy
            strategy = tf.distribute.MirroredStrategy()

            with ((strategy.scope())):
                # Define and compile your model within the strategy scope
                print("Training defocus model")
                start_time = time()
                model_defocus = DeepDefocusMultiOutputModel(width=input_size[0], height=input_size[1]
                                                            ).getFullModel(learning_rate=LEARNING_RATE_DEF,
                                                                               defocus_scaler=scaler, cs=2.7e7,
                                                                               kV=300)
                #model_defocus = DeepDefocusSimpleModel(width=input_size[0], height=input_size[1]
                #                                            ).getFullModel(learning_rate=LEARNING_RATE_DEF,
                #                                                           defocus_scaler=scaler, cs=2.7e7,
                #                                                           kV=300)

            # Train the model using fit method
            history_defocus = model_defocus.fit(traingen,
                                                validation_data=valgen,
                                                epochs=EPOCHS,
                                                callbacks=callbacks_list_def,
                                                verbose=1)



            elapsed_time = time() - start_time
            print("Time in training model: %0.10f seconds." % elapsed_time)

            if plots_Bool:
                make_training_plots(history_defocus, path_logs_defocus, "defocus_")


            one_image_data_path = df_test.head(1)['FILE'].values[0]
            # extract_CNN_layer_features(path_logs_defocus, one_image_data_path,
            #                             layers=6, defocus_scaler=scaler,
            #                             objective_res=objective_res, sampling_rate=sampling_rate, xDim=input_size[0])

        except Exception as e:
            print(e)
            exit('Problems with the training')
    else:
        print("No GPU devices available.")
        exit('Problems with the training')

    # -------------------------------- SAVING DEFOCUS MODEL AND VAL INFORMATION ---------------------------------------

    # TODO: NOT FOR THE MOMENT
    # model.save(os.path.join(modelDir, 'model.h5'))

    # -------------------------------------- TESTING DEFOCUS MODEL -----------------------------------------------------

    if testing_Bool:
        print("Test mode")
        # loadModelDir = os.path.join(modelDir, 'model.h5')
        # model = load_model(loadModelDir)
        model_defocus = DeepDefocusMultiOutputModel(width=input_size[0], height=input_size[1]).getFullModel(learning_rate=0.001,
                                                                                        defocus_scaler=scaler,
                                                                                       cs=2.7e7, kV=200)
        #model_defocus = DeepDefocusSimpleModel(width=input_size[0], height=input_size[1]).getFullModel(
        #    learning_rate=0.001,
        #    defocus_scaler=scaler,
        #    cs=2.7e7, kV=200)

        model_defocus.load_weights(filepath=os.path.join(path_logs_defocus, 'Best_Weights.weights.h5'))
        imagesTest, defocusTest, anglesTest = prepare_test_data(df_test)

        print("Testing defocus model")
        defocusPrediction_scaled = model_defocus.predict(testgen)
        # Transform back
        defocusPrediction = np.zeros_like(defocusPrediction_scaled)
        defocusPrediction[:, 0] = scaler.inverse_transform(defocusPrediction_scaled[:, 0].reshape(-1, 1)).flatten()
        defocusPrediction[:, 1] = scaler.inverse_transform(defocusPrediction_scaled[:, 1].reshape(-1, 1)).flatten()
        #defocusPrediction[:, 0] = defocusPrediction_scaled[:, 0] * scaler_factor
        #defocusPrediction[:, 1] = defocusPrediction_scaled[:, 1] * scaler_factor
        defocusPrediction[:, 2] = defocusPrediction_scaled[:, 2] * 180

        mae_u = mean_absolute_error(defocusTest[:, 0], defocusPrediction[:, 0])
        print("Final mean absolute error defocus_U val_loss: ", mae_u)

        mae_v = mean_absolute_error(defocusTest[:, 1], defocusPrediction[:, 1])
        print("Final mean absolute error defocus_V val_loss: ", mae_v)

        mae_a = mean_absolute_error(anglesTest[:, 0], defocusPrediction[:, 2])
        print("Final mean absolute error angle val_loss: ", mae_a)

        if ground_Truth:
            # Compute the absolute errors
            absolute_errors_U = np.abs(df_test[COLUMNS['defocus_U']] - df_test['DEFOCUS_U_Est'])
            absolute_errors_V = np.abs(df_test[COLUMNS['defocus_V']] - df_test['DEFOCUS_V_Est'])
            absolute_errors_A = np.abs(df_test[COLUMNS['angle']] - df_test['Angle_Est'])
            # Compute the Mean Absolute Error (MAE)
            mae_u_est = np.mean(absolute_errors_U)
            mae_v_est = np.mean(absolute_errors_V)
            mae_a_est = np.mean(absolute_errors_A)
            print("Final mean absolute error defocus_U_est val_loss: ", mae_u_est)
            print("Final mean absolute error defocus_V_est val_loss: ", mae_v_est)
            print("Final mean absolute error angle_est val_loss: ", mae_a_est)

        mae_test_path = os.path.join(path_logs_defocus, "mae_test_results.txt")

        with open(mae_test_path, "w") as f:
            f.write("Final mean absolute error defocus_U val_loss: {}\n".format(mae_u))
            f.write("Final mean absolute error defocus_V val_loss: {}\n".format(mae_v))
            f.write("Final mean absolute error angle val_loss: {}\n".format(mae_a))
            if ground_Truth:
                f.write("Final mean absolute error defocus_U_est val_loss: {}\n".format(mae_u_est))
                f.write("Final mean absolute error defocus_V_est val_loss: {}\n".format(mae_v_est))
                f.write("Final mean absolute error angle_est val_loss: {}\n".format(mae_a_est))

        print("Results written to mae_test_results.txt")

        if plots_Bool:
            make_testing_plots(defocusPrediction, defocusTest, path_logs_defocus)
            make_testing_angle_plots(defocusPrediction[:, 2], anglesTest, path_logs_defocus)

    exit(0)
