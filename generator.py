import random
import time
import pandas as pd
import numpy as np
from dataloader import load_data


def generate_pcb_file(component_data, n_points=100, x_min=0, x_max=200, y_min=0, y_max=200):

    lineinfo = ''
    for index in range(n_points):
        component_index = random.randint(0, len(component_data) - 1)
        data = component_data.iloc[component_index]
        part, nozzle = data['part'], data['nz']
        lineinfo += 'R' + str(index + 1) + '\t'
        pos_x, pos_y = np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)

        lineinfo += '{:.3f}'.format(pos_x) + '\t' + '{:.3f}'.format(
            pos_x) + '\t0.000\t0.000\t' + part + '\t\tA\t1-0 ' + nozzle + '\t1\t1\t1\t1\t1\t1\t1\tN\tL0\n'
    filepath = 'rd' + time.strftime('%d%H%M', time.localtime()) + '.txt'
    with open('data/' + filepath, 'w') as f:
        f.write(lineinfo)
    f.close()
    return filepath


if __name__ == '__main__':
    component_data = pd.DataFrame(pd.read_csv(filepath_or_buffer='component.txt', sep='\t', header=None))
    component_data.columns = ["part", "desc", "fdr", "nz", 'camera', 'group', 'feeder-limit', 'points']
    filepath = generate_pcb_file(component_data)

    pcb_data, component_data = load_data(filepath, feeder_limit=1, auto_register=1)

