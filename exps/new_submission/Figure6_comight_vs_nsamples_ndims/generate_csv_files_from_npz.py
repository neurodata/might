from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd


n_dims_2_ = 6


def make_csv_over_nsamples(
    root_dir,
    model_names,
    SIMULATIONS_NAMES,
    n_samples_list,
    n_dims_1,
    n_repeats,
):
    # generate CSV file for varying n-samples models
    results = defaultdict(list)
    for sim_name in SIMULATIONS_NAMES:
        for model_name in model_names:
            for n_samples in n_samples_list:
                for idx in range(n_repeats):
                    output_fname = (
                        root_dir
                        / "output"
                        / model_name
                        / sim_name
                        / f"{sim_name}_{n_samples}_{n_dims_1}_{n_dims_2_}_{idx}.npz"
                    )
                    # print(output_fname)
                    # print(output_fname.exists())

                    # print(output_fname.exists())
                    # Load data from the compressed npz file
                    try:
                        loaded_data = np.load(output_fname)
                    except Exception as e:
                        print(e, output_fname)
                        continue

                    # Extract variables with the same names
                    idx = loaded_data["idx"]
                    n_samples = loaded_data["n_samples"]
                    n_dims_1 = loaded_data["n_dims_1"]
                    sas98 = loaded_data["sas98"]
                    sim_name = loaded_data["sim_type"]
                    # threshold = loaded_data["threshold"]

                    results["idx"].append(idx)
                    results["n_samples"].append(n_samples)
                    results["n_dims_1"].append(n_dims_1)
                    results["sas98"].append(sas98)
                    results["sim_type"].append(sim_name)
                    results["model"].append(model_name)
                    # results["threshold"].append(threshold)

    df = pd.DataFrame(results)

    # Melt the DataFrame to reshape it
    df_melted = pd.melt(
        df,
        id_vars=["n_samples", "sim_type", "model"],
        value_vars=["sas98"],
        var_name="metric",
        value_name="metric_value",
    )

    # Convert "sim_type" to categorical type
    df_melted["sim_type"] = df_melted["sim_type"].astype(str)
    df_melted["model"] = df_melted["model"].astype(str)
    # df_melted["n_dims"] = df_melted["n_dims"].astype(int)
    df_melted["n_samples"] = df_melted["n_samples"].astype(int)
    df_melted["metric_value"] = df_melted["metric_value"].astype(float)
    return df_melted


def make_csv_over_ndims1(
    root_dir,
    model_names,
    SIMULATIONS_NAMES,
    n_dims_list,
    n_samples,
    n_repeats,
):
    # generate CSV file for varying n-samples models
    results = defaultdict(list)
    for sim_name in SIMULATIONS_NAMES:
        for model_name in model_names:
            for n_dims_1 in n_dims_list:
                for idx in range(n_repeats):
                    output_fname = (
                        root_dir
                        / "output"
                        / model_name
                        / sim_name
                        / f"{sim_name}_{n_samples}_{n_dims_1}_{n_dims_2_}_{idx}.npz"
                    )
                    # print(output_fname)
                    # print(output_fname.exists())

                    # print(output_fname.exists())
                    # Load data from the compressed npz file
                    try:
                        loaded_data = np.load(output_fname)
                    except Exception as e:
                        print(e, output_fname)
                        continue

                    # Extract variables with the same names
                    idx = loaded_data["idx"]
                    n_samples = loaded_data["n_samples"]
                    n_dims_1 = loaded_data["n_dims_1"]
                    sas98 = loaded_data["sas98"]
                    sim_name = loaded_data["sim_type"]
                    # threshold = loaded_data["threshold"]

                    results["idx"].append(idx)
                    results["n_samples"].append(n_samples)
                    results["n_dims_1"].append(n_dims_1)
                    results["sas98"].append(sas98)
                    results["sim_type"].append(sim_name)
                    results["model"].append(model_name)
                    # results["threshold"].append(threshold)

    df = pd.DataFrame(results)

    # Melt the DataFrame to reshape it
    df_melted = pd.melt(
        df,
        id_vars=["n_dims_1", "sim_type", "model"],
        value_vars=["sas98"],
        var_name="metric",
        value_name="metric_value",
    )

    # Convert "sim_type" to categorical type
    df_melted["sim_type"] = df_melted["sim_type"].astype(str)
    df_melted["model"] = df_melted["model"].astype(str)
    df_melted["n_dims_1"] = df_melted["n_dims_1"].astype(int)
    df_melted["metric_value"] = df_melted["metric_value"].astype(float)
    return df_melted


if __name__ == "__main__":
    root_dir = Path("/Volumes/Extreme Pro/cancer")
    SIMULATIONS_NAMES = ["mean_shiftv2"]

    model_names = [
        # "comight",
        # "might_viewone",
        # "might_viewtwo",
        # "might_viewoneandtwo",
        "knn",
        "knn_viewone",
        "knn_viewtwo",
    ]

    n_samples_list = [2**x for x in range(8, 13)]
    n_dims_1 = 4090
    n_repeats = 100
    print(n_samples_list)

    # save the dataframe to a csv file over n-samples
    df = make_csv_over_nsamples(
        root_dir, model_names, SIMULATIONS_NAMES, n_samples_list, n_dims_1, n_repeats
    )
    df.to_csv(root_dir / "output" / "results_vs_nsamples.csv", index=False)

    n_dims_list = [2**x - 6 for x in range(3, 13)]
    n_samples = 4096
    n_repeats = 100
    print(n_dims_list)

    # save the dataframe to a csv file over n-dims
    df = make_csv_over_ndims1(
        root_dir, model_names, SIMULATIONS_NAMES, n_dims_list, n_samples, n_repeats
    )
    df.to_csv(root_dir / "output" / "results_vs_ndims.csv", index=False)
