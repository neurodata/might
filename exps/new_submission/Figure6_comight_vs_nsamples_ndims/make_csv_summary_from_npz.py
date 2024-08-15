# Used for generating csv summary files for all model metrics
# from npz files.
#
# For specifically CoMIGHT power, use `make_csv_summary_for_comightpower_from_npz.py`.

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

n_dims_2_ = 6


def Calculate_SA98(y_true, y_pred_proba, max_fpr=0.02) -> float:
    if y_true.squeeze().ndim != 1:
        raise ValueError(f"y_true must be 1d, not {y_true.shape}")
    if 0 in y_true or -1 in y_true:
        fpr, tpr, thresholds = roc_curve(
            y_true, y_pred_proba[:, 1], pos_label=1, drop_intermediate=False
        )
    else:
        fpr, tpr, thresholds = roc_curve(
            y_true, y_pred_proba[:, 1], pos_label=2, drop_intermediate=False
        )
    s98 = max([tpr for (fpr, tpr) in zip(fpr, tpr) if fpr <= max_fpr])
    return s98


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
            n_samples_ = loaded_data["n_samples"]
            n_dims_1_ = loaded_data["n_dims_1"]
            sim_name = loaded_data["sim_type"]
            # threshold = loaded_data["threshold"]

            results["idx"].append(idx)
            results["n_samples"].append(n_samples_)
            results["n_dims_1"].append(n_dims_1_)
            results["sim_type"].append(sim_name)
            results["model"].append(model_name)
            if param_name == "sas98":
                # sas98 = loaded_data["sas98"]
                # results["sas98"].append(sas98)
                y_score = loaded_data["posterior_arr"]
                y_true = loaded_data["y"]
                assert len(y_score) == n_samples
                if "might" in model_name:
                    n_trees, n_samples, n_classes = y_score.shape
                    y_score_avg = np.nanmean(y_score, axis=0)
                    y_score_binary = y_score_avg  # [:, 1]

                    nan_rows = np.isnan(y_score_binary)
                    y_score_binary = y_score_binary[~nan_rows]
                    y_true = y_true[~nan_rows]
                else:
                    #     y_score_binary = y_score[:, -1]
                    y_score_binary = y_score
                s98 = Calculate_SA98(y_true, y_score_binary, max_fpr=0.02)
                results["sas98"].append(s98)
            elif param_name == "auc":
                y_score = loaded_data["posterior_arr"]
                y_true = loaded_data["y"]
                assert len(y_score) == n_samples
                if "might" in model_name:
                    n_trees, n_samples, n_classes = y_score.shape
                    y_score_avg = np.nanmean(y_score, axis=0)
                    y_score_binary = y_score_avg[:, 1]

                    nan_rows = np.isnan(y_score_binary)
                    y_score_binary = y_score_binary[~nan_rows]
                    y_true = y_true[~nan_rows]
                else:
                    y_score_binary = y_score[:, -1]
                auc = roc_auc_score(y_true, y_score_binary)
                results["auc"].append(auc)
            elif param_name == "cdcorr_pvalue":
                # print(dict(loaded_data).keys())
                cdcorr_pvalue = loaded_data["cdcorr_pvalue"]
                results["cdcorr_pvalue"].append(cdcorr_pvalue)
            elif param_name == "pvalue":
                pvalue = loaded_data["pvalue"]
                results["pvalue"].append(pvalue)
            elif param_name == "cmi":
                mi = loaded_data["cmi"]
                results["cmi"].append(mi)

                if "comight" in model_name:
                    try:
                        I_XZ_Y = loaded_data["I_XZ_Y"]
                        I_Z_Y = loaded_data["I_Z_Y"]
                        results["I_XZ_Y"].append(I_XZ_Y)
                        results["I_Z_Y"].append(I_Z_Y)
                    except Exception:
                        try:
                            I_XZ_Y = loaded_data["I_X1X2_Y"]
                            I_Z_Y = loaded_data["I_X1_Y"]
                            results["I_XZ_Y"].append(I_XZ_Y)
                            results["I_Z_Y"].append(I_Z_Y)
                        except Exception as e:
                            print(e)
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
            n_samples_ = loaded_data["n_samples"]
            n_dims_1_ = loaded_data["n_dims_1"]
            sim_name = loaded_data["sim_type"]
            # threshold = loaded_data["threshold"]

            results["idx"].append(idx)
            results["n_samples"].append(n_samples_)
            results["n_dims_1"].append(n_dims_1_)
            results["sim_type"].append(sim_name)
            results["model"].append(model_name)

            if param_name == "sas98":
                # sas98 = loaded_data["sas98"]
                # results["sas98"].append(sas98)
                y_score = loaded_data["posterior_arr"]
                y_true = loaded_data["y"]
                assert len(y_score) == n_samples
                if "might" in model_name:
                    n_trees, n_samples, n_classes = y_score.shape
                    y_score_avg = np.nanmean(y_score, axis=0)
                    y_score_binary = y_score_avg  # [:, 1]

                    nan_rows = np.isnan(y_score_binary)
                    y_score_binary = y_score_binary[~nan_rows]
                    y_true = y_true[~nan_rows]
                else:
                    #     y_score_binary = y_score[:, -1]
                    y_score_binary = y_score
                s98 = Calculate_SA98(y_true, y_score_binary, max_fpr=0.02)
                results["sas98"].append(s98)
            elif param_name == "cdcorr_pvalue":
                # print(dict(loaded_data).keys())
                cdcorr_pvalue = loaded_data["cdcorr_pvalue"]
                results["cdcorr_pvalue"].append(cdcorr_pvalue)
            elif param_name == "auc":
                y_score = loaded_data["posterior_arr"]
                y_true = loaded_data["y"]
                assert len(y_score) == n_samples
                if "might" in model_name:
                    n_trees, n_samples, n_classes = y_score.shape
                    y_score_avg = np.nanmean(y_score, axis=0)
                    y_score_binary = y_score_avg[:, 1]

                    nan_rows = np.isnan(y_score_binary)
                    y_score_binary = y_score_binary[~nan_rows]
                    y_true = y_true[~nan_rows]
                else:
                    y_score_binary = y_score[:, -1]
                auc = roc_auc_score(y_true, y_score_binary)
                results["auc"].append(auc)
            elif param_name == "cmi":
                mi = loaded_data["cmi"]
                results["cmi"].append(mi)

                if "comight" in model_name:
                    try:
                        I_XZ_Y = loaded_data["I_XZ_Y"]
                        I_Z_Y = loaded_data["I_Z_Y"]
                        results["I_XZ_Y"].append(I_XZ_Y)
                        results["I_Z_Y"].append(I_Z_Y)
                    except Exception:
                        try:
                            I_XZ_Y = loaded_data["I_X1X2_Y"]
                            I_Z_Y = loaded_data["I_X1_Y"]
                            results["I_XZ_Y"].append(I_XZ_Y)
                            results["I_Z_Y"].append(I_Z_Y)
                        except Exception as e:
                            print(e)
            # results["threshold"].append(threshold)

    df = pd.DataFrame(results)
    print(df.head())
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
    root_dir = Path("/Volumes/Extreme Pro/cancer/")
    # root_dir = Path('/home/hao/')
    # output_dir = Path('/data/adam/')
    output_dir = Path("/Volumes/Extreme Pro/cancer/temp/")

    n_repeats = 100

    n_samples_list = [2**x for x in range(8, 11)]
    n_dims_list = [2**x - 6 for x in range(3, 11)]
    # n_dims_1 = 1024 - 6
    n_dims_1 = 512 - 6
    # n_dims_1 = 4096 - 6
    print(n_samples_list)

    # Choose one of the parametr names
    # param_name = "cdcorr_pvalue"
    param_name = "cmi"
    param_name = "sas98"
    # param_name = 'auc'
    # param_name = "pvalue"

    if param_name == "sas98":
        models = [
            # "comight",
            # "comight-perm",
            "knn",
            "rf",
            "svm",
            "lr",
            # "knn_viewone",
            # "knn_viewtwo",
            #    'might_viewone', 'might_viewtwo'
        ]
    elif param_name == "auc":
        models = [
            "knn",
            "rf",
            "svm",
            "lr",
        ]
    elif param_name == "cmi":
        models = [
            "comight-cmi",
            "ksg",
        ]
    elif param_name == "cdcorr_pvalue":
        models = ["cdcorr"]
    # for model_name in ["coleman_pvalues"]:

    sim_names = ["mean_shiftv4", "multi_modalv2", "multi_equal"]
    for sim_name in sim_names:
        for model_name in models:
            n_dims_1 = 512 - 6
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
                / f"results_vs_nsamples_{sim_name}_{model_name}_{param_name}_{n_dims_1}_{n_repeats}.csv",
                index=False,
            )

            # Save the dataframe over varying ndims
            n_samples = 4096
            # n_samples = 1024
            n_samples = 512
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
                / f"results_vs_ndims_{sim_name}_{model_name}_{param_name}_{n_dims_1}_{n_repeats}.csv",
                index=False,
            )
