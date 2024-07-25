# What scripts generate what?

## Generate Data

The script `make_comight_datav2.py` is ran on the cluster, or locally to generate the dataset. This will result in 3 folders corresponding to the
simulation settings:

- `mean_shiftv4`: mean shift
- `multi_modalv2`: multi modal
- `multi_equal`: multi equal

## Running scripts locally

1. `comight_mi_vs_n_dims.py` will compute CoMIGHT's CMI estimate on simulations over varying n_dims
2. `comight_mi_vs_n_samples.py` will compute CoMIGHT's CMI estimate on simulations over varying n_samples
3. `comight_sa98_vs_ndims.py` will compute CoMIGHT's S@98 estimate on simulations over varying n_dims
4. `comight_sa98_vs_nsamples.py` will compute CoMIGHT's S@98 estimate on simulations over varying n_samples
5. `comight-perm_sa98_vs_nsamples_ndims.py` will compute a permutation test on CoMIGHT's S@98 estimate on simulations over varying n_samples and n_dims to allow us to produce a power curve.

The pvalue power curve compares the estimated statistics from the permutation test to the true statistics from the non-permuted forest.

## If running on PBS cluster

First, we need to generate a .txt file containing all the parametrizations we want to run.

We will need to run `generate_pbs_param_txt.py`: This will generate a `parameters_comight.txt` file, which contains four columns. For example:
`0 128 506 multi_modalv2`, which corresponds to:

1. index
2. n_samples
3. n_dims
4. dataset

Then one can submit PBS jobs for the following scripts, using the `pbs_submission_comight.sh` shell script:

- `comight_mi_cluster.py`
- `comight_sa98_cluster.py`
- `comight-perm_sa98_cluster.py`

Note: you will have to set the PBS parameters based on the cluster specifications, and the file paths in the shell script.

# Converting Saved NPZ files to CSV

`compute_metric_from_npz.py`