#%%
import rmm
from itertools import product

master_path = 'benchmarks/VF_benchmark_2'

# poles = [10, 15]
# widths = [1e-3]
# points = [6]
# ratios = [1e-3]

poles = [8, 10, 15]
widths = [1e-4, 1e-3, 1e-2, 1e-1]
points = [3 ,6, 30, 100]
ratios = [1e-4, 1e-3, 1e-2, 1e-1]

iterable = product(poles, widths, points, ratios)
tuples_file = open(master_path + '/arguments.txt', 'w')
tuples_file.write('Poles, Width, Points, Ratio\n')

for i, point in enumerate(iterable):
    print(i)
    tuples_file.write('Run '+ str(i) + ': ' + str(point) + '\n')

    path = master_path + "/run" + str(i)

    rmm.generate_data.main(path, 10, point[1], point[2], point[3])
    rmm.VF.main(path, 30, poles = point[0])

    args_file = open(path + '/arguments.txt', 'w')
    args_file.write("Width =" + str(point[0])+'\n')
    args_file.write("Points per pole =" + str(point[1])+'\n')
    args_file.write("Signal to noise ratio =" + str(point[2]))
    args_file.close()

tuples_file.close()

# %%
set(iterable)

# %%
