import os

n_repeats = 100
sim_name = "multi_modalv2"

curr_dir = os.getcwd()

with open(f"{curr_dir}/parameters_{sim_name}.txt", "w") as file:
    for n_dims_1 in [2**i - 6 for i in range(3, 13)]:
        for idx in range(n_repeats):
            # seed, n_samples, n_dims_1, sim_name
            params = [idx, 4096, n_dims_1, sim_name]
            file.write(" ".join(map(str, params)) + "\n")

    for n_samples in [2**x for x in range(8, 13)]:
        for idx in range(n_repeats):
            if n_samples != 4096 or n_dims_1 != 4090:
                params = [idx, n_samples, 4090, sim_name]
                file.write(" ".join(map(str, params)) + "\n")

print("Parameters file generated successfully.")
