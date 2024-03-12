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
    param_name,
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
                    sim_name = loaded_data["sim_type"]
                    # threshold = loaded_data["threshold"]

                    results["idx"].append(idx)
                    results["n_samples"].append(n_samples)
                    results["n_dims_1"].append(n_dims_1)
                    results["sim_type"].append(sim_name)
                    results["model"].append(model_name)
                    if param_name == "sas98":
                        sas98 = loaded_data["sas98"]
                        results["sas98"].append(sas98)
                    elif param_name == "cmi":
                        mi = loaded_data["cmi"]
                        results["cmi"].append(mi)

                        I_XZ_Y= loaded_data['I_XZ_Y']
                        I_Z_Y= loaded_data['I_Z_Y']
                        results['I_XZ_Y'].append(I_XZ_Y)
                        results['I_Z_Y'].append(I_Z_Y)
                    # results["threshold"].append(threshold)

    df = pd.DataFrame(results)

    # Melt the DataFrame to reshape it
    df_melted = pd.melt(
        df,
        id_vars=["n_samples", "sim_type", "model"],
        value_vars=[
            param_name
        ] if param_name == "sas98" else [param_name, 'I_XZ_Y', 'I_Z_Y'],
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
    param_name,
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
                    sim_name = loaded_data["sim_type"]
                    # threshold = loaded_data["threshold"]

                    results["idx"].append(idx)
                    results["n_samples"].append(n_samples)
                    results["n_dims_1"].append(n_dims_1)
                    results["sim_type"].append(sim_name)
                    results["model"].append(model_name)

                    if param_name == "sas98":
                        sas98 = loaded_data["sas98"]
                        results["sas98"].append(sas98)
                    elif param_name == "cmi":
                        mi = loaded_data["cmi"]
                        results["cmi"].append(mi)
                    # results["threshold"].append(threshold)

    df = pd.DataFrame(results)

    # Melt the DataFrame to reshape it
    df_melted = pd.melt(
        df,
        id_vars=["n_dims_1", "sim_type", "model"],
        value_vars=[param_name],
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
    # root_dir = Path('/data/sambit/')
    SIMULATIONS_NAMES = [
        # "mean_shiftv2", 
                         "multi_modalv2"]

    model_names = [
        "comight-cmi",
        # "comight-cmi-with-max09",
        # "might_viewone",
        # "might_viewtwo",
        # "might_viewoneandtwo",
        # "knn",
        # "knn_viewone",
        # "knn_viewtwo",
        # 'comight-cmi',
        # 'ksg',
    ]
    param_name = "cmi"

    n_samples_list = [2**x for x in range(8, 13)]
    n_dims_1 = 4090
    n_dims_1 = 1024 - 6
    # n_dims_1 = 16 - 6
    n_repeats = 10
    # n_repeats = 100
    print(n_samples_list)

    # save the dataframe to a csv file over n-samples
    df = make_csv_over_nsamples(
        root_dir,
        model_names,
        SIMULATIONS_NAMES,
        n_samples_list,
        n_dims_1,
        n_repeats,
        param_name=param_name,
    )
    df.to_csv(
        root_dir
        / "output"
        / f"results_vs_nsamples_{param_name}_{n_dims_1}_{n_repeats}.csv",
        index=False,
    )

    n_dims_list = [2**x - 6 for x in range(3, 13)]
    n_samples = 4096
    n_repeats = 100
    print(n_dims_list)

    # save the dataframe to a csv file over n-dims
    # df = make_csv_over_ndims1(
    #     root_dir, model_names, SIMULATIONS_NAMES, n_dims_list, n_samples, n_repeats
    # )
    # df.to_csv(root_dir / "output" / "results_vs_ndims.csv", index=False)
