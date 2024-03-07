n_repeats = 100

with open("parameters.txt", "w") as file:
    # for n_dims_1 in [2**i - 6 for i in range(3, 13)]:
    #     for sim_name in [
    #         'mean_shiftv2',
    #         # "mean_shift_compounding", "multi_modal_compounding", "multi_equal"
    #         ]:
    #         for idx in range(n_repeats):
    #             # seed, n_samples, n_dims_1, sim_name
    #             params = [idx, 4096, n_dims_1, sim_name]
    #             file.write(' '.join(map(str, params)) + '\n')

    for n_samples in [2**x for x in range(8, 13)]:
        for sim_name in [
            'mean_shiftv2',
            # "mean_shift_compounding", "multi_modal_compounding", "multi_equal"
            ]:
            for idx in range(n_repeats):
                # if n_samples != 4096 or n_dims_1 != 4090:
                params = [idx, n_samples, 4090, sim_name]
                file.write(' '.join(map(str, params)) + '\n')

print("Parameters file generated successfully.")
