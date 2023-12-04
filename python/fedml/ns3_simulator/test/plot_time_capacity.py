import matplotlib.pyplot as plt
import numpy as np

rows = 2
cols = 3
fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(24, 14))


underlay='geantdistance'
ps_num=9
model_size=2401488
core_link_capacity_list=[1000000000, 1000000000, 1000000000, 10000000, 100000000, 1000000000]
access_link_capacity_list=[10000000, 100000000, 1000000000, 1000000000, 1000000000, 1000000000]

def add_max_time(file_name, time_list):
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            data = line.split(",")
            if int(data[0]) == access_link_capacity and int(data[1]) == core_link_capacity :
                time_list.append(float(data[3]))

index = 0
for core_link_capacity, access_link_capacity in zip(core_link_capacity_list, access_link_capacity_list):
    row = index // cols
    col = index % cols
    ax =axs[row][col]
    index += 1

    # generate some random test data
    time_list = []
    alg_list = []
    patterns = ('/','//','-', '+', 'x', '\\', '\\\\', '*', 'o', 'O', '.')

    # read hfl
    file_name = '%s-%s-ps_%d-model_%d.txt' % ('hfl', underlay, ps_num, model_size)
    add_max_time(file_name, time_list)
    alg_list.append('HFL')

    # read rar
    file_name = '%s-%s-ps_%d-topo_%s-model_%d.txt' % ('rar', underlay, ps_num, 'ring', model_size)
    add_max_time(file_name, time_list)
    alg_list.append('RAR')

    # read pfl
    for topo in ['ring', '2d_torus', 'star', 'complete']:
        file_name = '%s-%s-ps_%d-topo_%s-model_%d.txt' % ('pfl', underlay, ps_num, topo, model_size)
        add_max_time(file_name, time_list)
        alg_list.append("PFL(%s)"%topo)

    # plot box plot
    ax.bar(alg_list, time_list, hatch=patterns[:len(alg_list)], color='white', edgecolor='black')
    ax.set_title('Core link=%d Mb, Access link=%d Mb' % (core_link_capacity/1e6, access_link_capacity/1e6))

    # adding horizontal grid lines
    ax.yaxis.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylabel('Communication time (seconds)', size=14)
    # ax.set_xlabel('Four separate samples')

plt.tight_layout()
plt.savefig("time_capacity.pdf", format="pdf", dpi=100, bbox_inches='tight')
# plt.show()