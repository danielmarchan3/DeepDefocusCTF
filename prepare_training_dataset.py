import os
import sys
import re
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.processing import rotate_image, sum_angles, compute_ctf_tf, process_micrographs_parallel

# Constants
DB_NAME = "ctfs.sqlite"
GROUND_TRUTH_PATH = "/home/dmarchan/data_hilbert_tres/TestNewPhantomData/simulationParameters.txt"


def load_ground_truth_values(filename: str) -> pd.DataFrame:
    """Load ground truth defocus values from a text file."""
    print("Loading ground truth defocus values")
    df = pd.read_csv(filename, sep=" ", header=None, index_col=False)
    df.columns = ["COUNTER", "FILE", "DEFOCUS_U", "DEFOCUS_V", "ANGLE"]
    return df


def get_defocus_angles_gt(df: pd.DataFrame, entry_fn: str) -> tuple:
    """Retrieve ground truth defocus values for a given filename."""
    entry_fn = os.path.basename(entry_fn)
    entry_fn = entry_fn.split("_ctf_xmipp")[0]
    found_entry = df[df["FILE"].str.contains(entry_fn)]

    if found_entry.empty:
        raise ValueError(f"Ground truth values not found for: {entry_fn}")

    return found_entry.iloc[0][["DEFOCUS_U", "DEFOCUS_V", "ANGLE"]]


def import_ctf(directory: str, use_ground_truth: bool) -> list:
    """Import CTF information and mic images from xmipp CTF estimation results or ground truth values."""
    file_list = []
    db_path = os.path.join(directory, DB_NAME)

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    print(f"Opening database: {db_path}")
    connection = sqlite3.connect(db_path)

    # Swap this line to adapt query for roodmus
    query = "SELECT id, enabled, c08, c01, c02, c03, c12 FROM Objects"
    # query = "SELECT id, enabled, c08, c01, c02, c03, c26, c12 FROM Objects"
    cursor = connection.execute(query)

    df_gt = load_ground_truth_values(GROUND_TRUTH_PATH) if use_ground_truth else None

    for row in cursor:
        # Swap this line to adapt query for roodmus
        ctf_id, enabled, file, dU, dV, dAngle, kV = row
        resolution = -1
        # ctf_id, enabled, file, dU, dV, dAngle, resolution, kV = row
        file = os.path.join(re.sub(r"/Runs.*", "", directory), file)

        if use_ground_truth:
            dU_estimated ,dV_estimated, dAngle_estimated = dU, dV, dAngle
            dU, dV, dAngle = get_defocus_angles_gt(df_gt, file)

        dSinA = np.sin(2 * dAngle)
        dCosA = np.cos(2 * dAngle)

        if use_ground_truth:
            entry = (ctf_id, dU, dV, dSinA, dCosA, dAngle, resolution, kV, file, dU_estimated, dV_estimated, dAngle_estimated)
        else:
            entry = (ctf_id, dU, dV, dSinA, dCosA, dAngle, resolution, kV, file)

        file_list.append(entry)

        print(f"Processed ID={ctf_id}, FILE={file}, DEFOCUS_U={dU}, DEFOCUS_V={dV}, ANGLE={dAngle}, kV={kV}")

    connection.close()
    print(f"Entries read from database: {len(file_list)}")
    print("Database closed successfully.")
    return file_list


def create_metadata_ctf(file_list: list, output_dir: str, pixel_size: float, use_ground_truth: bool):
    os.makedirs(output_dir, exist_ok=True)
    """
    Generate a metadata CSV file for training and copy files to the output directory.

    Parameters:
    - file_list (list): A list of tuples containing CTF information for each micrograph.
      Each tuple includes fields like ID, defocus values, angle, resolution, kV, and original file path.
    - output_dir (str): Directory where the metadata file and PSD files will be stored.
    - pixel_size (float): Original pixel size of the micrographs in Ångströms.
    - use_ground_truth (bool): If True, includes ground truth defocus values and angles in the metadata for comparison.

    Behavior:
    - Creates a DataFrame from file_list with columns for CTF parameters.
    - Adds a PIXEL_SIZE column.
    - Generates a unique PSD filename for each micrograph using the pattern:
      ID_basename_psd.npy and updates the FILE column to point to this PSD file.
    - Calls process_micrographs_parallel to compute PSDs for all micrographs.
    - Saves or appends the metadata to metadata.csv in output_dir.
    """

    if use_ground_truth:
        cols = ['ID', 'DEFOCUS_U', 'DEFOCUS_V', 'Sin(2*angle)', 'Cos(2*angle)', 'Angle', 'Resolution', 'kV', 'FILE',
                'DEFOCUS_U_Est', 'DEFOCUS_V_Est', 'Angle_Est']
    else:
        cols = ['ID', 'DEFOCUS_U', 'DEFOCUS_V', 'Sin(2*angle)', 'Cos(2*angle)', 'Angle', 'Resolution', 'kV', 'FILE']

    df_metadata = pd.DataFrame(file_list, columns=cols)
    df_metadata.insert(7, "PIXEL_SIZE", pixel_size, True)

    # Pair list (input_mrc_path, output_psd_path)
    mic_pairs = []

    for index in df_metadata.index:
        fn_root = df_metadata.loc[index, "FILE"]
        fn_base = os.path.basename(fn_root).replace(".mrc", "")
        ctf_id = df_metadata.loc[index, "ID"]
        psd_filename = os.path.join(output_dir, f"{ctf_id}_{fn_base}_psd.npy")

        # Save the output filename in the dataframe
        df_metadata.at[index, "FILE"] = psd_filename

        # Add pair to the processing list
        mic_pairs.append((fn_root, psd_filename))

    print(f"Number of micrographs to process: {len(mic_pairs)}")
    metadata_path = os.path.join(output_dir, "metadata.csv")

    # Send pairs to parallel processing function
    process_micrographs_parallel(mic_pairs, original_pixel_size=pixel_size, target_pixel_size=1)

    if os.path.exists(metadata_path):
        df_prev = pd.read_csv(metadata_path)
        df_metadata = pd.concat([df_prev, df_metadata], ignore_index=True)

    df_metadata.to_csv(metadata_path, index=False)
    print("Metadata file created successfully.")


def study_dataframe(df: pd.DataFrame, num_bins: int = 10) -> dict:
    """Analyze defocus distribution and determine additional cases needed per bin."""
    bins = pd.cut(df["DEFOCUS_U"], bins=num_bins)
    bin_counts = bins.value_counts()
    max_count_bin = bin_counts.idxmax()
    max_count = bin_counts[max_count_bin]

    additional_cases_needed = max_count - bin_counts
    return {
        f"interval_{i + 1}": {"left": interval.left, "right": interval.right, "extra_cases": extra_cases}
        for i, (interval, extra_cases) in enumerate(additional_cases_needed.items())
    }


def generate_phantom_data(output_dir: str):
    """Generate synthetic PSD images based on metadata file."""
    metadata_path = os.path.join(output_dir, "metadata.csv")

    if not os.path.exists(metadata_path):
        print("No metadata file found.")
        return

    df_metadata = pd.read_csv(metadata_path)

    for index, row in df_metadata.iterrows():
        file_path = row["FILE"]
        new_file_path = os.path.join(output_dir, os.path.basename(file_path))

        defocusU, defocusV, defocusA, kV = row["DEFOCUS_U"], row["DEFOCUS_V"], row["Angle"], row["kV"]
        cs = 2.7e7
        sampling_rate = 1
        size = 512

        print(f"Generating synthetic CTF: {new_file_path}")
        ctf_array = compute_ctf_tf(kV, sampling_rate, size, defocusU, defocusV, cs, 0, defocusA)
        np.save(new_file_path, ctf_array)

        df_metadata.at[index, "FILE"] = new_file_path

    df_metadata.to_csv(metadata_path, index=False)
    print("Phantom data generation completed.")


def rotate_and_update_entry(row, angle_rotation):
    """
    Rotates the image, updates the corresponding file name and angle values.
    """
    file_path = row['FILE']
    directory, file_name = os.path.split(file_path)

    rotated_image = rotate_image(file_path, angle_rotation)
    file_name_rot = f"{angle_rotation}_{file_name}"
    new_file_path = os.path.join(directory, file_name_rot)

    np.save(new_file_path, rotated_image)

    new_angle = sum_angles(row['Angle'], angle_rotation)

    return {
        'FILE': new_file_path,
        'Angle': new_angle,
        'Sin(2*angle)': np.sin(2 * new_angle),
        'Cos(2*angle)': np.cos(2 * new_angle)
    }


def augmentate_entries(entries_in_interval, number_extra_cases):
    """
    Augments data by rotating images and generating new entries.
    """
    if len(entries_in_interval) >= number_extra_cases:
        print('Number of entries in interval is greater than or equal to extra cases needed.')
        sampled_entries = entries_in_interval.sample(number_extra_cases)
    else:
        print('Extra cases exceed available entries.')
        sampled_entries = entries_in_interval.copy()

    augmented_entries = []

    for _, row in sampled_entries.iterrows():
        angle_steps = [90] if len(entries_in_interval) >= number_extra_cases else [45, 90]

        for angle in angle_steps:
            new_entry = rotate_and_update_entry(row, angle)
            augmented_entries.append({**row.to_dict(), **new_entry})

    return pd.DataFrame(augmented_entries)


def generate_data(dirOut):
    """
    Reads metadata, augments data by defocus intervals, and saves the updated dataset.
    """
    metadata_path = os.path.join(dirOut, "metadata.csv")

    if not os.path.exists(metadata_path):
        print('There is no metadata file.')
        return

    df = pd.read_csv(metadata_path)
    dict_intervals = study_dataframe(df, num_bins=15)

    df_defocus_1 = df[['DEFOCUS_U', 'DEFOCUS_V']]
    df_defocus_1.plot.hist(alpha=0.5, bins=15)
    plt.show()

    result_df = df.copy()

    for interval_data in dict_intervals.values():
        target_interval = (interval_data['left'], interval_data['right'])
        number_extra_cases = interval_data['extra_cases']

        if number_extra_cases == 0:
            continue

        entries_in_interval = df[(df['DEFOCUS_U'] >= target_interval[0]) & (df['DEFOCUS_U'] <= target_interval[1])]
        df_new_data = augmentate_entries(entries_in_interval, number_extra_cases)

        result_df = pd.concat([result_df, df_new_data], ignore_index=True)

    print('Final augmented dataset length:', len(result_df))

    df_defocus = result_df['DEFOCUS_U']
    df_defocus.plot.hist(alpha=0.5, bins=15)
    plt.show()

    result_df.to_csv(metadata_path, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python prepare_training_dataset.py <dirIn> <dirOut> <pixelSize> <increaseDataset(0|1|2)> <useGT(0|1)>")
        sys.exit(1)

    dir_in, dir_out = sys.argv[1], sys.argv[2]
    pixel_size = float(sys.argv[3])
    increase_dataset = int(sys.argv[4])
    ground_truth = int(sys.argv[5])

    use_ground_truth = True if ground_truth else False

    if increase_dataset == 0:
        # Normal functioning
        psd_files = import_ctf(dir_in, use_ground_truth=use_ground_truth)
        create_metadata_ctf(psd_files, dir_out, pixel_size, use_ground_truth=use_ground_truth)
    elif increase_dataset == 1:
        print("Balancing dataset with data augmentation")
        generate_data(dir_out)
    else:
        print("Generating phantom data with the CTF function")
        generate_phantom_data(dir_out)  # This will generate synthetic ctf function images

    sys.exit(0)
