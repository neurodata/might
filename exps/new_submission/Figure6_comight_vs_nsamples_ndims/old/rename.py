import os
import re


def rename_files(directory, old_number, new_number):
    # Regular expression pattern to match the desired filename pattern
    pattern = r"(.+?)_(\d+)_(\d+)_(\d+)_(\d+)\.npz"

    for filename in os.listdir(directory):
        # Check if the file matches the pattern
        if re.match(pattern, filename):
            # Extract parts of the filename
            match = re.match(pattern, filename)
            sim_name, n_samples, n_dims_1, n_dims_2, seed = match.groups()

            # Check if the n_dims_1 part matches the old_number
            if n_dims_1 == str(old_number):
                # Construct the new filename with updated n_dims_1
                new_n_dims_1 = str(new_number)
                new_filename = (
                    f"{sim_name}_{n_samples}_{new_n_dims_1}_{n_dims_2}_{seed}.npz"
                )

                # Construct full file paths
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)

                # Rename the file
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")


# Replace 'directory_path' with the path to your directory containing the files
directory_path = "/Volumes/Extreme Pro/cancer/output/cdcorr/multi_equal/"
old_number = 512  # Number to be replaced
new_number = 506  # Number to replace with
rename_files(directory_path, old_number, new_number)
