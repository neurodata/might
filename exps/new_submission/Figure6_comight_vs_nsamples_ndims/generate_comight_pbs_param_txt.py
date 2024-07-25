import os

n_repeats = 100
sim_names = ["multi_modalv2", "mean_shiftv4", "multi_equal"]

curr_dir = os.getcwd()

with open(f"{curr_dir}/parameters_comight.txt", "w") as file:
    for sim_name in sim_names:
        # for n_samples in [2**x for x in range(7, 11)]:
        #     for idx in range(n_repeats):
        #         params = [idx, n_samples, 512 - 6, sim_name]
        #         file.write(" ".join(map(str, params)) + "\n")

        for n_dims_1 in [2**i - 6 for i in range(3, 12)]:
            for idx in range(n_repeats):
                # seed, n_samples, n_dims_1, sim_name
                params = [idx, 4096, n_dims_1, sim_name]
                file.write(" ".join(map(str, params)) + "\n")

print("Parameters file generated successfully.")
