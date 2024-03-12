from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

n_dims_2_ = 6


def make_csv_over_nsamples(
    root_dir,
    sim_name,
    n_samples_list,
    n_dims_1,
    n_repeats,
    param_name,
    model_name,
):
    # generate CSV file for varying n-samples models
    results = defaultdict(list)
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

                if "comight" in model_name:
                    I_XZ_Y = loaded_data["I_XZ_Y"]
                    I_Z_Y = loaded_data["I_Z_Y"]
                    results["I_XZ_Y"].append(I_XZ_Y)
                    results["I_Z_Y"].append(I_Z_Y)
                # results["threshold"].append(threshold)

    df = pd.DataFrame(results)

    # Melt the DataFrame to reshape it
    df_melted = pd.melt(
        df,
        id_vars=["n_samples", "sim_type", "model"],
        value_vars=(
            [param_name]
            if param_name == "sas98" or "comight" not in model_name
            else [param_name, "I_XZ_Y", "I_Z_Y"]
        ),
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
    sim_name,
    n_dims_list,
    n_samples,
    n_repeats,
    param_name,
    model_name,
):
    # generate CSV file for varying n-samples models
    results = defaultdict(list)
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

                if "comight" in model_name:
                    I_XZ_Y = loaded_data["I_XZ_Y"]
                    I_Z_Y = loaded_data["I_Z_Y"]
                    results["I_XZ_Y"].append(I_XZ_Y)
                    results["I_Z_Y"].append(I_Z_Y)
            # results["threshold"].append(threshold)

    df = pd.DataFrame(results)

    # print('\n\n HERE!', param_name == "sas98" or  "comight" not in model_name)
    # Melt the DataFrame to reshape it
    df_melted = pd.melt(
        df,
        id_vars=["n_dims_1", "sim_type", "model"],
        value_vars=(
            [param_name]
            if param_name == "sas98" or "comight" not in model_name
            else [param_name, "I_XZ_Y", "I_Z_Y"]
        ),
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
    root_dir = Path('/home/hao/')
    output_dir = Path('/data/adam/output/')

    # sim_name = "multi_modal-5-102"
    sim_name = "mean_shiftv2"
    model_name = "might_viewone"
    model_name = "might_viewtwo"
    # model_name = "ksg"
    # model_name = 'knn_viewtwo'
    # model_name = 'knn'
    # model_name = 'comight-cmi'
    # perm_name = "comight-perm"
    # param_name = "sas98"
    param_name = "sas98"
    figname = "sas98"  # TODO: change

    n_samples_list = [2**x for x in range(8, 13)][-1:]
    n_dims_1 = 1024 - 6
    n_dims_1 = 4096 - 6
    n_repeats = 100
    print(n_samples_list)

    # save the dataframe to a csv file over n-samples
    df = make_csv_over_nsamples(
        root_dir,
        sim_name,
        n_samples_list,
        n_dims_1,
        n_repeats,
        param_name=param_name,
        model_name=model_name,
    )
    df.to_csv(
        output_dir
        / "output"
        / f"results_vs_nsamples_{figname}_{sim_name}_{model_name}_{param_name}_{n_dims_1}_{n_repeats}.csv",
        index=False,
    )

    # Save the dataframe over varying ndims
    n_dims_list = [2**x - 6 for x in range(3, 13)]
    n_samples = 4096
    n_repeats = 100
    print(n_dims_list)

    # save the dataframe to a csv file over n-dims
    df = make_csv_over_ndims1(
        root_dir,
        sim_name,
        n_dims_list,
        n_samples,
        n_repeats,
        param_name=param_name,
        model_name=model_name,
    )
    df.to_csv(
        output_dir
        / "output"
        / f"results_vs_ndims_{figname}_{sim_name}_{model_name}_{param_name}_{n_dims_1}_{n_repeats}.csv",
        index=False,
    )

    # perm_df = make_csv_over_ndims1(
    #     root_dir,
    #     SIMULATIONS_NAMES,
    #     n_dims_list,
    #     n_samples,
    #     n_repeats,
    #     param_name=param_name,
    #     model_name=perm_name,
    # )
    # perm_df.to_csv(
    #     root_dir
    #     / "output"
    #     / f"results_vs_ndims_{figname}_{sim_name}_{perm_name}_{param_name}_{n_dims_1}_{n_repeats}.csv",
    #     index=False,
    # )

    # perm_df = make_csv_over_nsamples(
    #     root_dir,
    #     sim_name,
    #     n_samples_list,
    #     n_dims_1,
    #     n_repeats,
    #     param_name=param_name,
    #     model_name=perm_name,
    # )
    # perm_df.to_csv(
    #     root_dir
    #     / "output"
    #     / f"results_vs_nsamples_{figname}_{sim_name}_{perm_name}_{param_name}_{n_dims_1}_{n_repeats}.csv",
    #     index=False,
    # )
