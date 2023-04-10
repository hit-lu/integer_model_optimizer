import copy
import gurobipy
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
        part_feederbase[idx] = slot     # part index - slot

    r = 1
    I, J = len(cpidx_2_part.keys()), len(nzidx_2_nozzle.keys())
    H = 6  # max head num

    # === determine the hyper-parameter of L ===
    # first phase: calculate the number of heads for each type of nozzle
    nozzle_heads = defaultdict(int)
    for nozzle in nozzle_list.keys():
        nozzle_heads[nozzle] = 1

    while sum(nozzle_heads.values()) != H:
        max_cycle_nozzle = None

        for nozzle, head_num in nozzle_heads.items():
            if max_cycle_nozzle is None or nozzle_list[nozzle] / head_num > nozzle_list[max_cycle_nozzle] / \
                    nozzle_heads[max_cycle_nozzle]:
                max_cycle_nozzle = nozzle

        assert max_cycle_nozzle is not None
        nozzle_heads[max_cycle_nozzle] += 1

    nozzle_comp_points = defaultdict(list)
    for part, points in component_list.items():
        idx = component_data[component_data['part'] == part].index.tolist()[0]
        nozzle = component_data.loc[idx]['nz']
        nozzle_comp_points[nozzle].append([part, points])

    level = 1 if len(component_list) == 1 and len(component_list) % H == 0 else 2
    part_assignment, cycle_assignment = [], []

    def aux_func(info):
        return max(map(lambda points: max([p[1] for p in points]), info))

    def recursive_assign(assign_points, nozzle_compo_points, cur_level, total_level) -> int:
        if cur_level > total_level:
            def func(points): return map(lambda points: max([p[1] for p in points]), points)
            return 0 if sum(func(nozzle_compo_points.values())) == 0 else 1

        if assign_points <= 0:
            if len(cycle_assignment) > 0:
                return 1
            return -1

        nozzle_compo_points_cpy = copy.deepcopy(nozzle_compo_points)
        part_cycle_assign = []
        for nozzle, head in nozzle_heads.items():
            while head:
                min_idx = -1
                for idx, (part, points) in enumerate(nozzle_compo_points_cpy[nozzle]):
                    if points >= assign_points and (
                            min_idx == -1 or points < nozzle_compo_points_cpy[nozzle][min_idx][1]):
                        min_idx = idx
                part_cycle_assign.append(-1 if min_idx == -1 else nozzle_compo_points_cpy[nozzle][min_idx][0])
                if min_idx != -1:
                    nozzle_compo_points_cpy[nozzle][min_idx][1] -= assign_points
                head -= 1

        part_assignment.append(part_cycle_assign)
        cycle_assignment.append(assign_points)
        res = recursive_assign(aux_func(nozzle_compo_points_cpy.values()), nozzle_compo_points_cpy,
                               cur_level + 1, total_level)
        if res == 0:
            return 0

        elif res == -1:
            # 当前周期分配点数为0，仍无法完成分配
            part_assignment.pop()
            cycle_assignment.pop()

            return recursive_assign(aux_func(nozzle_compo_points.values()), nozzle_compo_points, cur_level - 1,
                                    total_level)

        elif res == 1:
            # 所有周期均已走完，但是还有剩余的点未分配完
            part_assignment.pop()
            cycle_assignment.pop()
            return recursive_assign(assign_points - 1, nozzle_compo_points, cur_level, total_level)

    # second phase: (greedy) recursive search to assign points for each cycle set and obtain an initial solution
    while True:
        if recursive_assign(max(component_list.values()), nozzle_comp_points, 1, level) == 0:
            break
        level += 1

    weight_cycle, weight_nz_change, weight_pick = 2, 6, 1
    L = len(cycle_assignment)
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

    T = mdl.addVars(list_range(L), vtype=GRB.BINARY)

    if initial:
        # initial some variables to speed up the search process
        part_2_cpidx = defaultdict(int)
        for idx, part in cpidx_2_part.items():
            part_2_cpidx[part] = idx

        slot = 0
        for part, _ in sorted(component_list.items(), key=lambda x: x[0]):
            while slot in feeder_data.keys():
                slot += 1       # skip assigned feeder slot

            if part_2_cpidx[part] in part_feederbase.keys():
                continue

            part_feederbase[part_2_cpidx[part]] = slot
            slot += 1

        cycle_index = sorted(range(len(cycle_assignment)), key=lambda k: cycle_assignment[k], reverse=True)
        for idx, cycle in enumerate(cycle_index):
            WL[idx].Start = cycle_assignment[cycle]
            for h in range(H):
                part = part_assignment[cycle][h]
                if part == -1:
                    continue
                slot = part_feederbase[part_2_cpidx[part]]
                x[part_2_cpidx[part], slot, h, idx].Start = 1
                if type(f[slot, part_2_cpidx[part]]) == gurobipy.Var:
                    f[slot, part_2_cpidx[part]].Start = 1

    # === Objective ===
    mdl.modelSense = GRB.MINIMIZE
    mdl.setObjective(weight_cycle * quicksum(WL[l] for l in range(L)) + weight_nz_change * quicksum(
        NC[h] for h in range(H)) + weight_pick * quicksum(PU[s, l] for s in range(-(H - 1) * r, S) for l in range(L))
                     + 0.01 * quicksum(T[l] for l in range (L)))

    mdl.addConstrs(WL[l] <= M * T[l] for l in range(L))

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


def cal_individual_val(pcb_data, component_data, individual):
    feeder_data = defaultdict(int)
    for idx, slot in enumerate(individual):
        feeder_data[slot] = component_data.loc[idx]['part']
    objval, _, _, feeder_assign = gurobi_optimizer(pcb_data, component_data, feeder_data, initial=True, hinter=False)

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


def get_top_k_value(pop_val, k: int, reverse=True):
    res = []
    pop_val_cpy = copy.deepcopy(pop_val)
    pop_val_cpy.sort(reverse=reverse)

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
    crossover_rate, mutation_rate = 0.6, 0.1
    population_size, n_generations = 10, 200

    with tqdm(total=n_generations) as pbar:
        pbar.set_description('ILP genetic process')

        # initial solution
        population, pop_val = [], []
        for _ in range(population_size):
            pop_permutation = list_range(len(component_data))
            random.shuffle(pop_permutation)
            population.append(pop_permutation)
            pop_val.append(cal_individual_val(pcb_data, component_data, pop_permutation))

        best_individual, best_pop_val = [], []
        for _ in range(n_generations):
            # calculate fitness value
            pop_val = []
            for individual in population:
                pop_val.append(cal_individual_val(pcb_data, component_data, individual))

            best_individual = population[np.argmin(pop_val)]
            best_pop_val.append(min(pop_val))

            # min-max convert
            max_val = 1.5 * max(pop_val)
            sel_pop_val = list(map(lambda v: max_val - v, pop_val))

            # crossover and mutation
            new_population, new_pop_val = [], []
            for pop in range(population_size):
                if pop % 2 == 0 and random.random() < crossover_rate:
                    index1, index2 = roulette_wheel_selection(sel_pop_val), -1
                    while True:
                        index2 = roulette_wheel_selection(sel_pop_val)
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

                    new_pop_val.append(cal_individual_val(pcb_data, component_data, offspring1))
                    new_pop_val.append(cal_individual_val(pcb_data, component_data, offspring2))

            # selection
            top_k_index = get_top_k_value(pop_val, population_size - len(new_population), reverse=False)
            for index in top_k_index:
                new_population.append(population[index])
                new_pop_val.append(pop_val[index])

            population, pop_val = new_population, new_pop_val
            pbar.update(1)

    print(best_pop_val)
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
    # feeder_data = defaultdict(int)  # feeder arrangement slot-part
    #
    # feeder_test = list_range(len(component_data))
    #
    # random.shuffle(feeder_test)
    # for idx, slot in enumerate(feeder_test):
    #     feeder_data[slot] = component_data.loc[idx]['part']
    #
    # gurobi_optimizer(pcb_data, component_data, feeder_data, initial=True, hinter=True)
    genetic_algorithm(pcb_data, component_data)


if __name__ == '__main__':
    main()


