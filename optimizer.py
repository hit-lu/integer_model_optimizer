import copy
import math
import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import argparse

from ortools.sat.python import cp_model
from collections import defaultdict
from tqdm import tqdm

from dataloader import *
from gurobipy import *


class SolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)

    def on_solution_callback(self):
        print(self.ObjectiveValue())


def timer_wrapper(func):
    def func_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print('%s cost time: %.3f s' % (func_wrapper.__name__,  time.time() - start_time))
        return result
    return func_wrapper


def list_range(start, end=None):
    return list(range(start)) if end is None else list(range(start, end))


def gurobi_optimizer(pcb_data, component_data, feeder_data, initial=False, hinter=True):
    random.seed(0)
    # data preparation: convert data to index
    component_list, nozzle_list = defaultdict(int), defaultdict(int)
    cpidx_2_part, nzidx_2_nozzle, cpidx_2_nzidx = {}, {}, {}
    for _, data in pcb_data.iterrows():
        part = data['part']
        if part not in cpidx_2_part.values():
            cpidx_2_part[len(cpidx_2_part)] = part

        component_list[part] += 1

        idx = component_data[component_data['part'] == part].index.tolist()[0]
        nozzle = component_data.loc[idx]['nz']
        if nozzle not in nzidx_2_nozzle.values():
            nzidx_2_nozzle[len(nzidx_2_nozzle)] = nozzle
        nozzle_list[nozzle] += 1

        for cp_idx, part_ in cpidx_2_part.items():
            for nz_idx, nozzle_ in nzidx_2_nozzle.items():
                if part == part_ and nozzle == nozzle_:
                    cpidx_2_nzidx[cp_idx] = nz_idx
                    break

    part_feederbase = defaultdict(int)
    for slot, part in feeder_data.items():
        idx = -1
        for idx, part_ in cpidx_2_part.items():
            if part == part_:
                break
        assert idx != -1
        part_feederbase[idx] = slot

    weight_cycle, weight_nz_change, weight_pick = 2, 6, 1

    r = 1
    I, J = len(cpidx_2_part.keys()), len(nzidx_2_nozzle.keys())
    L = 2  # todo: how to deal with this hyper-parameter
    H = 6  # max head num
    S = r * I  # the available feeder num
    M = 1000  # a sufficient large number
    HC = [[0 for _ in range(J)] for _ in range(I)]
    for i in range(I):
        for cp_idx, part in cpidx_2_part.items():
            for j in range(J):
                HC[cp_idx][j] = 1 if cpidx_2_nzidx[cp_idx] == j else 0

    mdl = Model('SMT')
    mdl.setParam('Seed', 0)
    mdl.setParam('OutputFlag', hinter)          # set whether output the debug information

    # === Decision Variables ===
    x = mdl.addVars(list_range(I), list_range(S), list_range(H), list_range(L), vtype=GRB.BINARY)
    c = mdl.addVars(list_range(I), list_range(S), list_range(H), list_range(L), vtype=GRB.INTEGER,
                    ub=len(pcb_data) // H + 1)

    # todo: the condition for upper limits of feeders exceed 1
    f = {}
    for i in range(I):
        for s in range(S):
            if i in part_feederbase.keys():
                f[s, i] = 1 if part_feederbase[i] == s // r else 0
            else:
                f[s, i] = mdl.addVar(vtype=GRB.BINARY)

    p = mdl.addVars(list_range(-(H - 1) * r, S), list_range(L), vtype=GRB.BINARY)
    z = mdl.addVars(list_range(J), list_range(H), list_range(L), vtype=GRB.BINARY)
    d = mdl.addVars(list_range(L - 1), list_range(H), vtype=GRB.BINARY)
    d_plus = mdl.addVars(list_range(J), list_range(H), list_range(L - 1), vtype=GRB.BINARY)
    d_minus = mdl.addVars(list_range(J), list_range(H), list_range(L - 1), vtype=GRB.BINARY)

    PU = mdl.addVars(list_range(-(H - 1) * r, S), list_range(L), vtype=GRB.INTEGER)
    WL = mdl.addVars(list_range(L), vtype=GRB.INTEGER, ub=len(pcb_data) // H + 1)
    NC = mdl.addVars(list_range(H), vtype=GRB.INTEGER, ub=J)

    # initial process for speed up the search process
    if initial:
        # greedy heuristic initialization
        component_point_list = []
        for index, part in cpidx_2_part.items():
            component_point_list.append([index, component_list[part]])

        # first phase: ensure that each head has work
        while len(component_point_list) < H:
            component_point_list = sorted(component_point_list, key=lambda x: x[1], reverse=False)

            if component_point_list[-1][1] == 1:
                break

            component_point_list.append(component_point_list[-1].copy())
            component_point_list[-1][1] //= 2
            component_point_list[-2][1] -= component_point_list[-1][1]

        nozzle_points_list = defaultdict(list)
        for index, point in component_point_list:
            cp_idx = component_data[component_data['part'] == cpidx_2_part[index]].index.tolist()[0]
            nozzle = component_data.loc[cp_idx]['nz']
            nozzle_points_list[nozzle].append([index, point])

        for nozzle in nozzle_points_list.keys():
            # sort with the number of placement points
            nozzle_points_list[nozzle] = sorted(nozzle_points_list[nozzle], reverse=True, key=lambda x: x[1])

        # second phase: assignment process - ensure that each head has work
        cycle_idx, head_idx = 0, 0
        component_assignment = [[] for _ in range(H)]
        for nozzle, comp_point in nozzle_points_list.items():
            component_assignment[head_idx] = comp_point[0]
            nozzle_points_list[nozzle].pop(0)
            head_idx += 1

        while head_idx < H:
            max_point_nozzle, max_points = None, 0
            for nozzle, comp_point in nozzle_points_list.items():
                if len(comp_point) != 0 and comp_point[0][1] > max_points:
                    max_points, max_point_nozzle = comp_point[0][1], nozzle

            assert max_point_nozzle is not None
            component_assignment[head_idx] = nozzle_points_list[max_point_nozzle][0]
            nozzle_points_list[max_point_nozzle].pop(0)
            head_idx += 1

        # third phase assignment: assign other component points and calculate the initial result
        # condition1: all component points are assigned to the heads
        # condition2: the final result is determined
        while sum(map(len, nozzle_points_list.values())) != 0 or sum(map(lambda x: x[1], component_assignment)) != 0:
            workload = M
            for head_idx in range(H):
                if component_assignment[head_idx][1] == 0:
                    continue
                workload = min(component_assignment[head_idx][1], workload)
            assert workload != M

            WL[cycle_idx].Start = workload
            for head_idx in range(H):
                if component_assignment[head_idx][1] != 0:
                    cp_idx = component_assignment[head_idx][0]
                    nz_idx = cpidx_2_nzidx[cp_idx]
                    # nozzle_idx = component_data[]
                    # the initial slot for component is same to its index
                    x[cp_idx, cp_idx, head_idx, cycle_idx].Start = 1
                    z[nz_idx, head_idx, cycle_idx].Start = 1

                component_assignment[head_idx][1] -= workload

            if sum(map(len, nozzle_points_list.values())) != 0:
                while sum(map(lambda x: x[1], component_assignment)) != 0:
                    max_point_nozzle, max_points = None, 0
                    for nozzle, comp_point in nozzle_points_list.items():
                        if len(comp_point) != 0 and comp_point[0][1] > max_points:
                            max_points, max_point_nozzle = comp_point[0][1], nozzle

                    assert max_point_nozzle is not None
                    for head_idx in range(H):
                        if component_assignment[head_idx][1] == 0:
                            component_assignment[head_idx] = nozzle_points_list[max_point_nozzle][0]
                            nozzle_points_list[max_point_nozzle].pop(0)
                            break

    # === Objective ===
    mdl.modelSense = GRB.MINIMIZE
    mdl.setObjective(weight_cycle * quicksum(WL[l] for l in range(L)) + weight_nz_change * quicksum(
        NC[h] for h in range(H)) + weight_pick * quicksum(PU[s, l] for s in range(-(H - 1) * r, S) for l in range(L)))

    # === Constraint ===
    # work completion
    mdl.addConstrs(
        c[i, s, h, l] <= M * x[i, s, h, l] for i in range(I) for s in range(S) for h in range(H) for l in range(L))
    mdl.addConstrs(c[i, s, h, l] <= WL[l] for i in range(I) for s in range(S) for h in range(H) for l in range(L))
    mdl.addConstrs(
        c[i, s, h, l] >= WL[l] - M * (1 - x[i, s, h, l]) for i in range(I) for s in range(S) for h in range(H) for l in
        range(L))

    # variable constraint
    mdl.addConstrs(quicksum(x[i, s, h, l] for i in range(I)) <= 1 for s in range(S) for h in range(H) for l in range(L))
    mdl.addConstrs(quicksum(x[i, s, h, l] for s in range(S)) <= 1 for i in range(I) for h in range(H) for l in range(L))

    mdl.addConstrs(
        quicksum(c[i, s, h, l] for s in range(S) for h in range(H) for l in range(L)) == component_list[cpidx_2_part[i]]
        for i in range(I))

    # simultaneous pick
    for s in range(-(H - 1) * r, S):
        rng = list(range(max(0, -math.floor(s / r)), min(H, math.ceil((S - s) / r))))
        for l in range(L):
            mdl.addConstr(quicksum(x[i, s + h * r, h, l] for h in rng for i in range(I)) <= M * p[s, l], name='')
            mdl.addConstr(quicksum(x[i, s + h * r, h, l] for h in rng for i in range(I)) >= p[s, l], name='')

    mdl.addConstrs(PU[s, l] <= M * p[s, l] for s in range(-(H - 1) * r, S) for l in range(L))
    mdl.addConstrs(PU[s, l] <= WL[l] for s in range(-(H - 1) * r, S) for l in range(L))
    mdl.addConstrs(PU[s, l] >= WL[l] - M * (1 - p[s, l]) for s in range(-(H - 1) * r, S) for l in range(L))

    # nozzle change
    mdl.addConstrs(
        z[j, h, l] - z[j, h, l + 1] == d_plus[j, h, l] - d_minus[j, h, l] for l in range(L - 1) for j in range(J) for h
        in range(H))

    mdl.addConstrs(
        2 * d[l, h] == quicksum(d_plus[j, h, l] for j in range(J)) + quicksum(d_minus[j, h, l] for j in range(J)) for l
        in range(L - 1) for h in range(H))

    mdl.addConstrs(NC[h] == quicksum(d[l, h] for l in range(L - 1)) for h in range(H))

    # nozzle-component compatibility
    mdl.addConstrs(
        quicksum(x[i, s, h, l] for s in range(S)) <= quicksum(HC[i][j] * z[j, h, l] for j in range(J)) for i in
        range(I) for h in range(H) for l in range(L))

    # available number of feeder
    mdl.addConstrs(quicksum(f[s, i] for s in range(S)) <= 1 for i in range(I))

    # available number of nozzle
    mdl.addConstrs(quicksum(z[j, h, l] for h in range(H)) <= H for j in range(J) for l in range(L))

    # upper limit for occupation for feeder slot
    mdl.addConstrs(quicksum(f[s, i] for i in range(I)) <= 1 for s in range(S))

    # others
    mdl.addConstrs(quicksum(z[j, h, l] for j in range(J)) <= 1 for h in range(H) for l in range(L))
    mdl.addConstrs(
        quicksum(x[i, s, h, l] for h in range(H) for l in range(L)) >= f[s, i] for i in range(I) for s in range(S))
    mdl.addConstrs(
        quicksum(x[i, s, h, l] for h in range(H) for l in range(L)) <= M * f[s, i] for i in range(I) for s in
        range(S))

    # the constraints to speed up the search process
    mdl.addConstrs(quicksum(z[j, h, l] for j in range(J) for h in range(H)) >= quicksum(
        z[j, h, l + 1] for j in range(J) for h in range(H)) for l in range(L - 1))
    mdl.addConstr(quicksum(WL[l] for l in range(L)) >= len(pcb_data) // H, name='')
    mdl.addConstrs(WL[l] >= WL[l + 1] for l in range(L - 1))
    mdl.addConstrs(quicksum(x[i, s, h, l] for i in range(I) for s in range(S)) <= 1 for h in range(H) for l in range(L))

    # === search process ===
    mdl.update()
    mdl.optimize()

    # === result generation ===
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    columns = ['H{}'.format(i + 1) for i in range(H)] + ['cycle']

    nozzle_assign = pd.DataFrame(columns=columns)
    component_assign = pd.DataFrame(columns=columns)
    feeder_assign = pd.DataFrame(columns=columns)

    if mdl.Status == GRB.OPTIMAL or mdl.Status == GRB.INTERRUPTED:
        for l in range(L):
            if abs(WL[l].x) <= 1e-10:
                continue

            nozzle_assign.loc[l, 'cycle'] = WL[l].x
            component_assign.loc[l, 'cycle'] = WL[l].x
            feeder_assign.loc[l, 'cycle'] = WL[l].x

            for h in range(H):
                assigned = False
                for i in range(I):
                    for s in range(S):
                        if abs(x[i, s, h, l].x - 1) < 1e-10:
                            assigned = True
                            component_assign.loc[l, 'H{}'.format(h + 1)] = cpidx_2_part[i]
                            # component_assign.loc[l, 'H{}'.format(h + 1)] = 'CP' + str(i)
                            feeder_assign.loc[l, 'H{}'.format(h + 1)] = 'F' + str(s + 1)

                for j in range(J):
                    if abs(z[j, h, l].x - 1) < 1e-10:
                        nozzle_assign.loc[l, 'H{}'.format(h + 1)] = nzidx_2_nozzle[j]

                if not assigned:
                    nozzle_assign.loc[l, 'H{}'.format(h + 1)] = 'A'
                    component_assign.loc[l, 'H{}'.format(h + 1)] = ''
                    feeder_assign.loc[l, 'H{}'.format(h + 1)] = ''

        if hinter:
            print('total cost = {}'.format(mdl.objval))
            print('cycle = {}, nozzle change = {}, pick up = {}'.format(quicksum(WL[l].x for l in range(L)),
                                                                        quicksum(NC[h].x for h in range(H)), quicksum(
                    PU[s, l].x for s in range(-(H - 1) * r, S) for l in range(L))))

            print('workload: ')
            for l in range(L):
                print(WL[l].x, end=', ')

            print('')
            print('pick up information')
            for l in range(L):
                print('level ' + str(l + 1), ': ', end='')
                for s in range(-(H - 1) * r, S):
                    print(PU[s, l].x, end=', ')
                print('')

            print('')
            print('result')
            print(nozzle_assign)
            print(component_assign)
            print(feeder_assign)

    return mdl.objval if mdl.Status == GRB.OPTIMAL else 0, nozzle_assign, component_assign, feeder_assign


@timer_wrapper
def cal_individual_val(pcb_data, component_data, individual):
    feeder_data = defaultdict(int)
    for idx, slot in enumerate(individual):
        feeder_data[slot] = component_data.loc[idx]['part']
    objval, _, _, feeder_assign = gurobi_optimizer(pcb_data, component_data, feeder_data, hinter=False)

    # 增加移动距离的约束
    pick_distance = 0
    for _, data in feeder_assign.iterrows():
        slot_list = []
        for h in range(6):
            if data['H' + str(h + 1)] == '':
                continue
            slot = int(data['H' + str(h + 1)][1:]) - h
            slot_list.append(slot)

        sorted(slot_list)
        for idx in range(len(slot_list) - 1):
            pick_distance += abs(slot_list[idx + 1] - slot_list[idx])
        pick_distance *= data['cycle']

    return objval + 0.1 * pick_distance


def roulette_wheel_selection(pop_eval):
    # Roulette wheel
    cumsum_pop_eval = np.array(pop_eval)
    cumsum_pop_eval = np.divide(cumsum_pop_eval, np.sum(cumsum_pop_eval))
    cumsum_pop_eval = cumsum_pop_eval.cumsum()

    random_eval = np.random.random()
    index = 0
    while index < len(pop_eval):
        if random_eval > cumsum_pop_eval[index]:
            index += 1
        else:
            break
    return index


def swap_mutation(parent):
    range_ = np.random.randint(0, len(parent), 2)
    parent[range_[0]], parent[range_[1]] = parent[range_[1]], parent[range_[0]]
    return parent


def get_top_k_value(pop_val, k: int):
    res = []
    pop_val_cpy = copy.deepcopy(pop_val)
    pop_val_cpy.sort(reverse=True)

    for i in range(min(len(pop_val_cpy), k)):
        for j in range(len(pop_val)):
            if abs(pop_val_cpy[i] - pop_val[j]) < 1e-9 and j not in res:
                res.append(j)
                break
    return res


def partially_mapped_crossover(parent1, parent2):
    range_ = np.random.randint(0, len(parent1), 2)      # 前闭后开
    range_ = sorted(range_)

    parent1_cpy, parent2_cpy = [-1 for _ in range(len(parent1))], [-1 for _ in range(len(parent2))]

    parent1_cpy[range_[0]: range_[1] + 1] = copy.deepcopy(parent2[range_[0]: range_[1] + 1])
    parent2_cpy[range_[0]: range_[1] + 1] = copy.deepcopy(parent1[range_[0]: range_[1] + 1])

    for index in range(len(parent1)):
        if range_[0] <= index <= range_[1]:
            continue

        cur_ptr, cur_elem = 0, parent1[index]
        while True:
            parent1_cpy[index] = cur_elem
            if parent1_cpy.count(cur_elem) == 1:
                break
            parent1_cpy[index] = -1

            if cur_ptr == 0:
                cur_ptr, cur_elem = 1, parent2[index]
            else:
                index_ = parent1_cpy.index(cur_elem)
                cur_elem = parent2[index_]

    for index in range(len(parent2)):
        if range_[0] <= index <= range_[1]:
            continue

        cur_ptr, cur_elem = 0, parent2[index]
        while True:
            parent2_cpy[index] = cur_elem
            if parent2_cpy.count(cur_elem) == 1:
                break
            parent2_cpy[index] = -1

            if cur_ptr == 0:
                cur_ptr, cur_elem = 1, parent1[index]
            else:
                index_ = parent2_cpy.index(cur_elem)
                cur_elem = parent1[index_]

    return parent1_cpy, parent2_cpy


def genetic_algorithm(pcb_data, component_data):
    # basic parameter
    # crossover rate & mutation rate: 80% & 10%
    # population size: 30
    # the number of generation: 100
    crossover_rate, mutation_rate = 0.8, 0.1
    population_size, n_generations = 20, 10

    # initial solution
    population = []
    for _ in range(population_size):
        pop_permutation = list_range(len(component_data))
        random.shuffle(pop_permutation)
        population.append(pop_permutation)
    best_individual, best_pop_val = [], []

    with tqdm(total=n_generations) as pbar:
        pbar.set_description('ILP genetic process')

        for _ in range(n_generations):
            # calculate fitness value
            pop_val = []
            for individual in population:
                pop_val.append(cal_individual_val(pcb_data, component_data, individual))

            best_individual = population[np.argmin(pop_val)]
            best_pop_val.append(min(pop_val))

            # min-max convert
            max_val = 1.5 * max(pop_val)
            pop_val = list(map(lambda v: max_val - v, pop_val))

            # crossover and mutation
            new_population = []
            for pop in range(population_size):
                if pop % 2 == 0 and random.random() < crossover_rate:
                    index1, index2 = roulette_wheel_selection(pop_val), -1
                    while True:
                        index2 = roulette_wheel_selection(pop_val)
                        if index1 != index2:
                            break

                    # 两点交叉算子
                    offspring1, offspring2 = partially_mapped_crossover(population[index1], population[index2])

                    if np.random.random() < mutation_rate:
                        swap_mutation(offspring1)

                    if np.random.random() < mutation_rate:
                        swap_mutation(offspring2)

                    new_population.append(offspring1)
                    new_population.append(offspring2)

            # selection
            top_k_index = get_top_k_value(pop_val, population_size - len(new_population))
            for index in top_k_index:
                new_population.append(population[index])

            population = new_population
            pbar.update(1)

    plt.plot(best_pop_val)
    plt.show()

    feeder_data = defaultdict(int)
    for idx, slot in enumerate(best_individual):
        feeder_data[slot] = component_data.loc[idx]['part']

    _, nozzle_assign, component_assign, feeder_assign = gurobi_optimizer(pcb_data, component_data, feeder_data,
                                                                         hinter=True)

    return nozzle_assign, component_assign, feeder_assign


@timer_wrapper
def main():
    parser = argparse.ArgumentParser(description='integer programming optimizer implementation')
    parser.add_argument('--filename', default='PCB.txt', type=str, help='load pcb data')
    parser.add_argument('--auto_register', default=1, type=int, help='register the component according the pcb data')
    params = parser.parse_args()

    # 结果输出显示所有行和列
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # 加载PCB数据
    pcb_data, component_data, feeder_data = load_data(params.filename, default_feeder_limit=1,
                                                      cp_auto_register=params.auto_register)  # 加载PCB数据
    # 构造飞达数据
    feeder_data = defaultdict(int)  # feeder arrangement slot-part
    # random.seed(0)
    #
    # feeder_test = list_range(len(component_data))
    # random.shuffle(feeder_test)
    # for idx, slot in enumerate(feeder_test):
    #     feeder_data[slot] = component_data.loc[idx]['part']

    gurobi_optimizer(pcb_data, component_data, feeder_data, initial=False, hinter=True)
    # genetic_algorithm(pcb_data, component_data)


if __name__ == '__main__':
    main()


