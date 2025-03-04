import matplotlib.pyplot as plt
import os
import numpy as np


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