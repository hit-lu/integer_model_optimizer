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

# 机器参数
max_head_index, max_slot_index = 6, 120
interval_ratio = 2
slot_interval = 15
head_interval = slot_interval * interval_ratio

# 位置信息
slotf1_pos, slotr1_pos = [-31.267, 44.], [807., 810.545]  # F1(前基座最左侧)、R1(后基座最右侧)位置
fix_camera_pos = [269.531, 694.823]  # 固定相机位置
anc_marker_pos = [336.457, 626.230]  # ANC基准点位置
stopper_pos = [635.150, 124.738]  # 止档块位置

# 时间参数
t_pick, t_place = .078, .051  # 贴装/拾取用时
t_nozzle_put, t_nozzle_pick = 0.9, 0.75  # 装卸吸嘴用时
t_fix_camera_check = 0.12  # 固定相机检测时间

# 电机参数
head_rotary_velocity = 8e-5  # 贴装头R轴旋转时间
x_max_velocity, y_max_velocity = 1.4, 1.2
x_max_acceleration, y_max_acceleration = x_max_velocity / 0.079, y_max_velocity / 0.079


def axis_moving_time(distance, axis=0):
    distance = abs(distance) * 1e-3
    Lamax = x_max_velocity ** 2 / x_max_acceleration if axis == 0 else y_max_velocity ** 2 / y_max_acceleration
    Tmax = x_max_velocity / x_max_acceleration if axis == 0 else y_max_velocity / y_max_acceleration
    if axis == 0:
        return 2 * math.sqrt(distance / x_max_acceleration) if distance < Lamax else 2 * Tmax + (
                    distance - Lamax) / x_max_velocity
    else:
        return 2 * math.sqrt(distance / y_max_acceleration) if distance < Lamax else 2 * Tmax + (
                    distance - Lamax) / y_max_velocity


def head_rotary_time(angle):
    while -180 > angle > 180:
        if angle > 180:
            angle -= 360
        else:
            angle += 360
    return abs(angle) * head_rotary_velocity


def timer_wrapper(func):
    def func_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print('%s cost time: %.3f s' % (func_wrapper.__name__, time.time() - start_time))
        return result

    return func_wrapper


def list_range(start, end=None):
    return list(range(start)) if end is None else list(range(start, end))


def gurobi_optimizer(pcb_data, component_data, feeder_data, initial=False, hinter=True):
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
    if feeder_data:
        for slot, part in feeder_data.items():
            idx = -1
            for idx, part_ in cpidx_2_part.items():
                if part == part_:
                    break
            assert idx != -1
            part_feederbase[idx] = slot  # part index - slot

    r = 1
    I, J = len(cpidx_2_part.keys()), len(nzidx_2_nozzle.keys())

    # === determine the hyper-parameter of L ===
    # first phase: calculate the number of heads for each type of nozzle
    nozzle_heads = defaultdict(int)
    for nozzle in nozzle_list.keys():
        nozzle_heads[nozzle] = 1

    while sum(nozzle_heads.values()) != max_head_index:
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

    level = 1 if len(component_list) == 1 and len(component_list) % max_head_index == 0 else 2
    part_assignment, cycle_assignment = [], []

    def aux_func(info):
        return max(map(lambda points: max([p[1] for p in points]), info))

    def recursive_assign(assign_points, nozzle_compo_points, cur_level, total_level) -> int:
        def func(points):
            return map(lambda points: max([p[1] for p in points]), points)

        if cur_level > total_level and sum(func(nozzle_compo_points.values())) == 0:
            return 0
        elif assign_points <= 0 and cur_level == 1:
            return -1  # backtrack
        elif assign_points <= 0 or cur_level > total_level:
            return 1  # fail

        nozzle_compo_points_cpy = copy.deepcopy(nozzle_compo_points)
        prev_assign = 0
        for part in part_assignment[cur_level - 1]:
            if part != -1:
                prev_assign += 1

        head_idx = 0
        for nozzle, head in nozzle_heads.items():
            while head:
                min_idx = -1
                for idx, (part, points) in enumerate(nozzle_compo_points_cpy[nozzle]):
                    if points >= assign_points and (
                            min_idx == -1 or points < nozzle_compo_points_cpy[nozzle][min_idx][1]):
                        min_idx = idx
                part_assignment[cur_level - 1][head_idx] = -1 if min_idx == -1 else \
                    nozzle_compo_points_cpy[nozzle][min_idx][0]
                if min_idx != -1:
                    nozzle_compo_points_cpy[nozzle][min_idx][1] -= assign_points
                head -= 1
                head_idx += 1

        cycle_assignment[cur_level - 1] = assign_points
        for part in part_assignment[cur_level - 1]:
            if part != -1:
                prev_assign -= 1

        if prev_assign == 0:
            res = 1
        else:
            points = min(len(pcb_data) // max_head_index + 1, aux_func(nozzle_compo_points_cpy.values()))
            res = recursive_assign(points, nozzle_compo_points_cpy, cur_level + 1, total_level)
        if res == 0:
            return 0
        elif res == 1:
            # All cycles have been completed, but there are still points left to be allocated
            return recursive_assign(assign_points - 1, nozzle_compo_points, cur_level, total_level)

    # second phase: (greedy) recursive search to assign points for each cycle set and obtain an initial solution
    while True:
        part_assignment = [[-1 for _ in range(max_head_index)] for _ in range(level)]
        cycle_assignment = [-1 for _ in range(level)]
        points = min(len(pcb_data) // max_head_index + 1, max(component_list.values()))
        if recursive_assign(points, nozzle_comp_points, 1, level) == 0:
            break
        level += 1

    weight_cycle, weight_nz_change, weight_pick = 2, 6, 1

    L = len(cycle_assignment)
    S = r * I  # the available feeder num
    M = len(pcb_data)  # a sufficiently large number (number of placement points)
    HC = [[0 for _ in range(J)] for _ in range(I)]
    for i in range(I):
        for j in range(J):
            HC[i][j] = 1 if cpidx_2_nzidx[i] == j else 0

    mdl = Model('SMT')
    mdl.setParam('Seed', 0)
    mdl.setParam('OutputFlag', hinter)  # set whether output the debug information
    # mdl.setParam('TimeLimit', 3)
    # mdl.setParam('MIPFocus', 2)
    # mdl.setParam("Heuristics", 0.5)

    # Use only if other methods, including exploring the tree with the default settings, do not yield a viable solution
    # mdl.setParam("ZeroObjNodes", 100)

    # === Decision Variables ===
    y = mdl.addVars(list_range(I), list_range(max_head_index), list_range(L), vtype=GRB.BINARY, name='y')
    w = mdl.addVars(list_range(S), list_range(max_head_index), list_range(L), vtype=GRB.BINARY, name='w')

    c = mdl.addVars(list_range(I), list_range(max_head_index), list_range(L), vtype=GRB.INTEGER,
                    ub=len(pcb_data) // max_head_index + 1, name='c')

    # todo: the condition for upper limits of feeders exceed 1
    f = {}
    for i in range(I):
        for s in range(S):
            if i in part_feederbase.keys():
                f[s, i] = 1 if part_feederbase[i] == s // r else 0
            else:
                f[s, i] = mdl.addVar(vtype=GRB.BINARY, name='f_' + str(s) + '_' + str(i))

    p = mdl.addVars(list_range(-(max_head_index - 1) * r, S), list_range(L), vtype=GRB.BINARY, name='p')
    z = mdl.addVars(list_range(J), list_range(max_head_index), list_range(L), vtype=GRB.BINARY)

    d = mdl.addVars(list_range(L - 1), list_range(max_head_index), vtype=GRB.CONTINUOUS, name='d')
    d_plus = mdl.addVars(list_range(J), list_range(max_head_index), list_range(L - 1), vtype=GRB.CONTINUOUS,
                         name='d_plus')
    d_minus = mdl.addVars(list_range(J), list_range(max_head_index), list_range(L - 1), vtype=GRB.CONTINUOUS,
                          name='d_minus')

    PU = mdl.addVars(list_range(-(max_head_index - 1) * r, S), list_range(L), vtype=GRB.INTEGER, name='PU')
    WL = mdl.addVars(list_range(L), vtype=GRB.INTEGER, ub=len(pcb_data) // max_head_index + 1, name='WL')
    NC = mdl.addVars(list_range(max_head_index), vtype=GRB.CONTINUOUS, name='NC')
    PT = mdl.addVars(list_range(L), vtype=GRB.BINARY, name='PT')  # pick-and-place task

    if initial:
        # initial some variables to speed up the search process
        part_2_cpidx = defaultdict(int)
        for idx, part in cpidx_2_part.items():
            part_2_cpidx[part] = idx

        part_list = []
        for cycle_part in part_assignment:
            for part in cycle_part:
                if part != -1 and part not in part_list:
                    part_list.append(part)
        slot = 0
        for part in part_list:
            if feeder_data:
                while slot in feeder_data.keys():
                    slot += 1  # skip assigned feeder slot

            if part_2_cpidx[part] in part_feederbase.keys():
                continue

            part_feederbase[part_2_cpidx[part]] = slot
            slot += 1

        # ensure the priority of the workload assignment
        cycle_index = sorted(range(len(cycle_assignment)), key=lambda k: cycle_assignment[k], reverse=True)
        for idx, cycle in enumerate(cycle_index):
            WL[idx].Start = cycle_assignment[cycle]
            # for h in range(max_head_index):
            #     part = part_assignment[cycle][h]
            #     if part == -1:
            #         continue
            #     slot = part_feederbase[part_2_cpidx[part]]
            #     x[part_2_cpidx[part], slot, h, idx].Start = 1
            #     if type(f[slot, part_2_cpidx[part]]) == gurobipy.Var:
            #         f[slot, part_2_cpidx[part]].Start = 1

    # === Objective ===
    mdl.setObjective(weight_cycle * quicksum(WL[l] for l in range(L)) + weight_nz_change * quicksum(
        NC[h] for h in range(max_head_index)) + weight_pick * quicksum(
        PU[s, l] for s in range(-(max_head_index - 1) * r, S) for l in range(L)) + 0.01 * quicksum(
        PT[l] for l in range(L)))

    # === Constraint ===
    # work completion
    mdl.addConstrs(c[i, h, l] == WL[l] * y[i, h, l] for i in range(I) for h in range(max_head_index) for l in range(L))
    mdl.addConstrs(
        quicksum(c[i, h, l] for h in range(max_head_index) for l in range(L)) == component_list[cpidx_2_part[i]] for i
        in range(I))

    mdl.addConstrs(WL[l] <= M * PT[l] for l in range(L))

    # variable constraint
    mdl.addConstrs(
        quicksum(y[i, h, l] * w[s, h, l] for i in range(I) for s in range(S)) <= 1 for h in range(max_head_index) for l
        in range(L))

    mdl.addConstrs(
        quicksum(WL[l] * y[i, h, l] for h in range(max_head_index) for l in range(L)) == component_list[cpidx_2_part[i]]
        for i in range(I))

    # simultaneous pick
    for s in range(-(max_head_index - 1) * r, S):
        rng = list(range(max(0, -math.floor(s / r)), min(max_head_index, math.ceil((S - s) / r))))
        for l in range(L):
            mdl.addConstr(quicksum(w[s + h * r, h, l] for h in rng) <= M * p[s, l])
            mdl.addConstr(quicksum(w[s + h * r, h, l] for h in rng) >= p[s, l])

    mdl.addConstrs(PU[s, l] == p[s, l] * WL[l] for s in range(-(max_head_index - 1) * r, S) for l in range(L))
    # nozzle change
    mdl.addConstrs(
        z[j, h, l] - z[j, h, l + 1] == d_plus[j, h, l] - d_minus[j, h, l] for l in range(L - 1) for j in range(J) for h
        in range(max_head_index))

    mdl.addConstrs(
        2 * d[l, h] == quicksum(d_plus[j, h, l] for j in range(J)) + quicksum(d_minus[j, h, l] for j in range(J)) for l
        in range(L - 1) for h in range(max_head_index))

    mdl.addConstrs(NC[h] == quicksum(d[l, h] for l in range(L - 1)) for h in range(max_head_index))

    # nozzle-component compatibility
    mdl.addConstrs(
        y[i, h, l] <= quicksum(HC[i][j] * z[j, h, l] for j in range(J)) for i in range(I) for h in range(max_head_index)
        for l in range(L))

    # available number of feeder
    mdl.addConstrs(quicksum(f[s, i] for s in range(S)) <= 1 for i in range(I))

    # available number of nozzle
    mdl.addConstrs(quicksum(z[j, h, l] for h in range(max_head_index)) <= max_head_index for j in range(J) for l in range(L))

    # upper limit for occupation for feeder slot
    mdl.addConstrs(quicksum(f[s, i] for i in range(I)) <= 1 for s in range(S))
    mdl.addConstrs(
        quicksum(w[s, h, l] for s in range(S)) >= quicksum(y[i, h, l] for i in range(I)) for h in range(max_head_index)
        for l in range(L))

    # others
    mdl.addConstrs(quicksum(z[j, h, l] for j in range(J)) <= 1 for h in range(max_head_index) for l in range(L))
    mdl.addConstrs(
        quicksum(y[i, h, l] * w[s, h, l] for h in range(max_head_index) for l in range(L)) >= f[s, i] for i in range(I)
        for s in range(S))
    mdl.addConstrs(
        quicksum(y[i, h, l] * w[s, h, l] for h in range(max_head_index) for l in range(L)) <= M * f[s, i] for i in
        range(I) for s in range(S))

    # the constraints to speed up the search process
    mdl.addConstrs(quicksum(z[j, h, l] for j in range(J) for h in range(max_head_index)) >= quicksum(
        z[j, h, l + 1] for j in range(J) for h in range(max_head_index)) for l in range(L - 1))

    mdl.addConstr(quicksum(WL[l] for l in range(L)) >= math.ceil(len(pcb_data) // max_head_index))
    mdl.addConstrs(WL[l] >= WL[l + 1] for l in range(L - 1))

    # === search process ===
    mdl.update()
    # mdl.write('mdl.lp')
    print('num of constrs: ', str(len(mdl.getConstrs())), ', num of vars: ', str(len(mdl.getVars())))
    mdl.optimize()

    # === result generation ===
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    columns = ['H{}'.format(i + 1) for i in range(max_head_index)] + ['cycle']

    nozzle_assign = pd.DataFrame(columns=columns)
    component_assign = pd.DataFrame(columns=columns)
    feeder_assign = pd.DataFrame(columns=columns)

    if mdl.Status == GRB.OPTIMAL or mdl.Status == GRB.INTERRUPTED or GRB.TIME_LIMIT:
        for l in range(L):
            if abs(WL[l].x) <= 1e-10:
                continue

            nozzle_assign.loc[l, 'cycle'] = WL[l].x
            component_assign.loc[l, 'cycle'] = WL[l].x
            feeder_assign.loc[l, 'cycle'] = WL[l].x

            for h in range(max_head_index):
                assigned = False
                for i in range(I):
                    if abs(y[i, h, l].x - 1) < 1e-10:
                        assigned = True
                        component_assign.loc[l, 'H{}'.format(h + 1)] = cpidx_2_part[i]
                        # component_assign.loc[l, 'H{}'.format(h + 1)] = 'CP' + str(i)
                        # feeder_assign.loc[l, 'H{}'.format(h + 1)] = 'F' + str(s + 1)

                        for j in range(J):
                            if HC[i][j]:
                                nozzle_assign.loc[l, 'H{}'.format(h + 1)] = nzidx_2_nozzle[j]

                for s in range(S):
                    if abs(w[s, h, l].x - 1) < 1e-10:
                        feeder_assign.loc[l, 'H{}'.format(h + 1)] = 'F' + str(s + 1)

                if not assigned:
                    nozzle_assign.loc[l, 'H{}'.format(h + 1)] = ''
                    component_assign.loc[l, 'H{}'.format(h + 1)] = ''
                    feeder_assign.loc[l, 'H{}'.format(h + 1)] = ''

        if hinter:
            print('total cost = {}'.format(mdl.objval))
            print('cycle = {}, nozzle change = {}, pick up = {}'.format(quicksum(WL[l].x for l in range(L)), quicksum(
                NC[h].x for h in range(max_head_index)), quicksum(
                PU[s, l].x for s in range(-(max_head_index - 1) * r, S) for l in range(L))))

            print('workload: ')
            for l in range(L):
                print(WL[l].x, end=', ')

            print('')
            print('result')
            print(nozzle_assign)
            print(component_assign)
            print(feeder_assign)

    return mdl, nozzle_assign, component_assign, feeder_assign


def cal_individual_val(pcb_data, component_data, individual):
    feeder_data = defaultdict(int)
    for idx, slot in enumerate(individual):
        feeder_data[slot] = component_data.loc[idx]['part']
    mdl, _, _, feeder_assign = gurobi_optimizer(pcb_data, component_data, feeder_data, initial=True, hinter=False)

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

    return mdl.objval + 0.1 * pick_distance if mdl.Status == GRB.OPTIMAL else 0


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
    range_ = np.random.randint(0, len(parent1), 2)  # 前闭后开
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
    population_size, n_generations = 20, 50
    with tqdm(total=n_generations) as pbar:
        pbar.set_description('ILP genetic process')

        # initial solution
        population, pop_val = [], []
        for _ in range(population_size):
            pop_permutation = list_range(len(component_data))
            random.shuffle(pop_permutation)
            population.append(pop_permutation)

            # calculate fitness value
            pop_val.append(cal_individual_val(pcb_data, component_data, pop_permutation))

        best_pop_val = []
        for _ in range(n_generations):
            best_pop_val.append(min(pop_val))

            print(best_pop_val[-1])
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
    best_individual = population[np.argmin(pop_val)]
    for idx, slot in enumerate(best_individual):
        feeder_data[slot] = component_data.loc[idx]['part']

    _, nozzle_assign, component_assign, feeder_assign = gurobi_optimizer(pcb_data, component_data, feeder_data,
                                                                         hinter=True)

    return nozzle_assign, component_assign, feeder_assign


def dynamic_programming_cycle_path(pcb_data, cycle_placement, assigned_feeder):
    head_sequence = []
    num_pos = sum([placement != -1 for placement in cycle_placement]) + 1

    pos, head_set = [], []
    feeder_set = set()
    for head, feeder in enumerate(assigned_feeder):
        if feeder is None or cycle_placement[head] == -1:
            continue

        head_set.append(head)
        placement = cycle_placement[head]

        pos.append([pcb_data.loc[placement]['x'] - head * head_interval + stopper_pos[0],
                    pcb_data.loc[placement]['y'] + stopper_pos[1]])

        feeder_set.add(feeder - head * 1)   # todo: ratio 暂时设为1
    # todo: 供料器位置未确定 此处取平均值临时代替
    # pos.insert(0, [slotf1_pos[0] + ((min(list(feeder_set)) + max(list(feeder_set))) / 2 - 1) * slot_interval,
    #                slotf1_pos[1]])
    pos.insert(0, [sum(map(lambda x: x[0], pos)) / len(pos), slotf1_pos[1]])

    def get_distance(pos_1, pos_2):
        return math.sqrt((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2)

    # 各节点之间的距离
    dist = [[get_distance(pos_1, pos_2) for pos_2 in pos] for pos_1 in pos]

    min_dist = [[np.inf for _ in range(num_pos)] for s in range(1 << num_pos)]
    min_path = [[[] for _ in range(num_pos)] for s in range(1 << num_pos)]

    # 状压dp搜索
    for s in range(1, 1 << num_pos, 2):
        # 考虑节点集合s必须包括节点0
        if not (s & 1):
            continue
        for j in range(1, num_pos):
            # 终点j需在当前考虑节点集合s内
            if not (s & (1 << j)):
                continue
            if s == int((1 << j) | 1):
                # 若考虑节点集合s仅含节点0和节点j，dp边界，赋予初值
                # print('j:', j)
                min_path[s][j] = [j]
                min_dist[s][j] = dist[0][j]

            # 枚举下一个节点i，更新
            for i in range(1, num_pos):
                # 下一个节点i需在考虑节点集合s外
                if s & (1 << i):
                    continue
                if min_dist[s][j] + dist[j][i] < min_dist[s | (1 << i)][i]:
                    min_path[s | (1 << i)][i] = min_path[s][j] + [i]
                    min_dist[s | (1 << i)][i] = min_dist[s][j] + dist[j][i]

    ans_dist = float('inf')
    ans_path = []
    # 求最终最短哈密顿回路
    for i in range(1, num_pos):
        if min_dist[(1 << num_pos) - 1][i] + dist[i][0] < ans_dist:
            # 更新，回路化
            ans_path = min_path[s][i]
            ans_dist = min_dist[(1 << num_pos) - 1][i] + dist[i][0]

    for parent in ans_path:
        head_sequence.append(head_set[parent - 1])

    start_head, end_head = head_sequence[0], head_sequence[-1]
    if pcb_data.loc[cycle_placement[start_head]]['x'] - start_head * head_interval > \
            pcb_data.loc[cycle_placement[end_head]]['x'] - end_head * head_interval:
        head_sequence = list(reversed(head_sequence))
    return head_sequence


def greedy_placement_route_generation(component_data, pcb_data, component_assign, feeder_slot_assign):
    placement_result, head_sequence_result = [], []
    mount_point_index = [[] for _ in range(len(component_data))]
    mount_point_pos = [[] for _ in range(len(component_data))]
    for i in range(len(pcb_data)):
        part = pcb_data.loc[i]['part']
        component_index = component_data[component_data['part'] == part].index.tolist()[0]
        # 记录贴装点序号索引和对应的位置坐标
        mount_point_index[component_index].append(i)
        mount_point_pos[component_index].append([pcb_data.loc[i]['x'], pcb_data.loc[i]['y']])

    search_dir = 1  # 0：自左向右搜索  1：自右向左搜索
    for cycle_index, component_cycle in component_assign.iterrows():
        for _ in range(int(component_cycle['cycle'])):
            # search_dir = 1 - search_dir
            assigned_placement = [-1] * max_head_index
            max_pos = [max(mount_point_pos[component_index], key=lambda x: x[0]) for component_index in
                       range(len(mount_point_pos)) if len(mount_point_pos[component_index]) > 0][0][0]
            min_pos = [min(mount_point_pos[component_index], key=lambda x: x[0]) for component_index in
                       range(len(mount_point_pos)) if len(mount_point_pos[component_index]) > 0][0][0]
            point2head_range = min(math.floor((max_pos - min_pos) / head_interval) + 1, max_head_index)

            # 最近邻确定
            way_point = None
            head_range = range(max_head_index - 1, -1, -1) if search_dir else range(max_head_index)
            for head_counter, head in enumerate(head_range):
                part = component_cycle['H' + str(head + 1)]
                if part == '':
                    continue

                component_index = component_data[component_data['part'] == part].index.tolist()[0]
                if way_point is None or head_counter % point2head_range == 0:
                    index = 0
                    if way_point is None:
                        if search_dir:
                            index = np.argmax(mount_point_pos[component_index], axis=0)[0]
                        else:
                            index = np.argmin(mount_point_pos[component_index], axis=0)[0]
                    else:
                        for next_head in head_range:
                            part = component_cycle['H' + str(next_head + 1)]
                            if assigned_placement[next_head] == -1 and part != '':
                                component_index = component_data[component_data['part'] == part].index.tolist()[0]
                                num_points = len(mount_point_pos[component_index])
                                index = np.argmin(
                                    [abs(mount_point_pos[component_index][i][0] - way_point[0]) * .1 + abs(
                                        mount_point_pos[component_index][i][1] - way_point[1]) for i in
                                     range(num_points)])
                                head = next_head
                                break
                    # index = np.argmax(mount_point_pos[component_index], axis=0)[0]
                    assigned_placement[head] = mount_point_index[component_index][index]

                    # 记录路标点
                    way_point = mount_point_pos[component_index][index]
                    way_point[0] += (max_head_index - head - 1) * head_interval if search_dir else -head * head_interval

                    mount_point_index[component_index].pop(index)
                    mount_point_pos[component_index].pop(index)

                else:
                    head_index, point_index = -1, -1
                    min_cheby_distance, min_euler_distance = float('inf'), float('inf')
                    for next_head in range(max_head_index):
                        part = component_cycle['H' + str(next_head + 1)]
                        if assigned_placement[next_head] != -1 or part == '':
                            continue
                        next_comp_index = component_data[component_data['part'] == part].index.tolist()[0]
                        for counter in range(len(mount_point_pos[next_comp_index])):
                            if search_dir:
                                delta_x = abs(mount_point_pos[next_comp_index][counter][0] - way_point[0]
                                              + (max_head_index - next_head - 1) * head_interval)
                            else:
                                delta_x = abs(mount_point_pos[next_comp_index][counter][0] - way_point[0]
                                              - next_head * head_interval)

                            delta_y = abs(mount_point_pos[next_comp_index][counter][1] - way_point[1])

                            euler_distance = pow(axis_moving_time(delta_x, 0), 2) + pow(axis_moving_time(delta_y, 1), 2)
                            cheby_distance = max(axis_moving_time(delta_x, 0),
                                                 axis_moving_time(delta_y, 1)) + 5e-2 * euler_distance
                            if cheby_distance < min_cheby_distance or (abs(cheby_distance - min_cheby_distance) < 1e-9
                                                                       and euler_distance < min_euler_distance):
                                # if euler_distance < min_euler_distance:
                                min_cheby_distance, min_euler_distance = cheby_distance, euler_distance
                                head_index, point_index = next_head, counter

                    part = component_cycle['H' + str(head_index + 1)]
                    component_index = component_data[component_data['part'] == part].index.tolist()[0]
                    assert (0 <= head_index < max_head_index)

                    assigned_placement[head_index] = mount_point_index[component_index][point_index]
                    way_point = mount_point_pos[component_index][point_index]
                    way_point[0] += (max_head_index - head_index - 1) * head_interval if search_dir \
                        else -head_index * head_interval

                    mount_point_index[component_index].pop(point_index)
                    mount_point_pos[component_index].pop(point_index)

            placement_result.append(assigned_placement)  # 各个头上贴装的元件类型
            feeder_slot = []
            for head in range(max_head_index):
                slot = feeder_slot_assign.loc[cycle_index]['H' + str(head + 1)]
                feeder_slot.append(None if slot == '' else int(slot[1:]) - head * 1)  # todo: 此处暂时设为1，未做兼容处理
            head_sequence_result.append(dynamic_programming_cycle_path(pcb_data, assigned_placement, feeder_slot))

    return placement_result, head_sequence_result


def eular(posA, posB, interval=0):
    return math.sqrt(pow(posA[0] - posB[0] - interval, 2) + pow(posA[1] - posB[1], 2))


def cheby(posA, posB, interval=0):
    return max(abs(posA[0] - posB[0] - interval), abs(posA[1] - posB[1]))


def cal_individual_route_val(pcb_data, component_data, individual):
    mount_point_index = [[] for _ in range(len(component_data))]
    mount_point_pos = [[] for _ in range(len(component_data))]
    for i in range(len(pcb_data)):
        part = pcb_data.loc[i]['part']
        component_index = component_data[component_data['part'] == part].index.tolist()[0]
        # 记录贴装点序号索引和对应的位置坐标
        mount_point_index[component_index].append(i)
        mount_point_pos[component_index].append([pcb_data.loc[i]['x'], pcb_data.loc[i]['y']])

    point_head = {}
    for idx, i in enumerate(individual):
        if idx < 4:
            point_head[i] = 0
        elif idx < 8:
            point_head[i] = 1
        elif idx < 11:
            point_head[i] = 2
        elif idx < 14:
            point_head[i] = 3
        elif idx < 17:
            point_head[i] = 4
        else:
            point_head[i] = 5
    # 假定初始位置
    deport = slotf1_pos
    deport[0] += 45 * slot_interval
    saving = []

    for i in individual:
        for j in individual:
            if i == j or point_head[i] == point_head[j]:
                continue
            # todo: 目前仅处理一种元件的情形
            distance = cheby(deport, mount_point_pos[0][i]) + cheby(deport, mount_point_pos[0][j])

            distance -= cheby(mount_point_pos[0][i], mount_point_pos[0][j],
                              (point_head[j] - point_head[i]) * head_interval)
            saving.append([i, j, distance])

    saving = sorted(saving, key=lambda x: x[2], reverse=True)
    routes, routes_head = [], []
    for i in individual:
        routes.append([i])
        routes_head.append([point_head[i]])

    for i in range(len(saving)):
        start_route, end_route = [], []
        start_route_head, end_route_head = [], []
        for j in range(len(routes)):
            if len(routes) == max_head_index:
                continue
            if saving[i][0] == routes[j][-1]:
                end_route = routes[j]
                end_route_head = routes_head[j]
            elif saving[i][1] == routes[j][0]:
                start_route = routes[j]
                start_route_head = routes_head[j]

            # todo: 如何同元件分配对应？
            if len(end_route) > 0 and len(start_route) > 0:
                route_store = end_route + start_route
                route_head_store = end_route_head + start_route_head

                if len(route_store) <= max_head_index and len(set(route_head_store)) == len(route_head_store):

                    start_idx = routes.index(start_route)
                    end_index = routes.index(end_route)

                    if start_idx < end_index:
                        start_idx, end_index = end_index, start_idx

                    routes.pop(start_idx)
                    routes.pop(end_index)
                    routes.append(route_store)

                    routes_head.pop(start_idx)
                    routes_head.pop(end_index)
                    routes_head.append(route_head_store)

                break

    for i in range(len(routes)):
        if mount_point_pos[0][routes[i][0]][0] > mount_point_pos[0][routes[i][-1]][0]:
            routes[i] = routes[i][::-1]
            routes_head[i] = routes_head[i][::-1]

    val = 0
    assigned_feeder = [45] * max_head_index

    for idx_route, route in enumerate(routes):
        prev_pos = None
        cycle_placement = [-1 for _ in range(max_head_index)]

        for idx, i in enumerate(route):
            head = routes_head[idx_route][idx]
            cycle_placement[head] = i

        head_seq = dynamic_programming_cycle_path(pcb_data, cycle_placement, assigned_feeder)

        for head in head_seq:
            i = cycle_placement[head]
            pos = mount_point_pos[0][i]
            pos[0] -= head * head_interval
            if prev_pos is not None:
                val += cheby(prev_pos, pos)
            prev_pos = pos

    print('route value:', val)
    return val, routes, routes_head


# ======================== 聚类相关 ========================
# 两点距离
def distance(e1, e2):
    return np.sqrt((e1[0]-e2[0])**2+(e1[1]-e2[1])**2)


# 集合中心
def means(arr):
    return np.array([np.mean([e[0] for e in arr]), np.mean([e[1] for e in arr])])


# arr中距离a最远的元素，用于初始化聚类中心
def farthest(k_arr, arr):
    f = None
    max_d = 0
    for e in arr:
        d = 0
        for i in range(k_arr.__len__()):
            d = d + np.sqrt(distance(k_arr[i], e))
        if d > max_d:
            max_d = d
            f = e
    return f


# arr中距离a最近的元素，用于聚类
def closest(a, arr):
    c = arr[1]
    min_d = distance(a, arr[1])
    arr = arr[1:]
    for e in arr:
        d = distance(a, e)
        if d < min_d:
            min_d = d
            c = e
    return c


def cal_individual_cluster_val(pcb_data, individual, component_assign, feeder_slot_assign):
    head_div = 7
    # todo: 单一元件贴装点数较多时的处理，多种类型元件的编码方式
    mount_pos = []
    for idx, point in enumerate(individual):
        mount_pos.append(
            [pcb_data.iloc[point]['x'] - (idx // head_div) * head_interval, pcb_data.iloc[point]['y'], idx // head_div,
             point])

    # 簇的数量
    m = head_div

    # 随机选一个点放入
    r = np.random.randint(len(mount_pos) - 1)
    k_arr = np.array([[mount_pos[r][0], mount_pos[r][1]]])

    # 按照距离最远的原则，再选m-1个点放入
    cla_arr = [[]]
    for i in range(m-1):
        k = farthest(k_arr, mount_pos)
        k_arr = np.concatenate([k_arr, np.array([[k[0], k[1]]])])
        cla_arr.append([])

    # 迭代聚类
    n = 10                      # todo: 此处暂时设定为该值，单次运行时间结果为3s，时长仍需压缩
    cla_temp = cla_arr
    cla_head = [[] for _ in range(m)]
    for i in range(n):    # 迭代n次
        for e in mount_pos:    # 把集合里每一个元素聚到最近的类
            ki = -1        # 假定距离第一个中心最近
            min_d = 100000000
            for j in range(k_arr.__len__()):
                # 找到更近的聚类中心
                if cla_temp[j].__len__() < max_head_index and e[2] not in cla_head[j] and distance(e, k_arr[j]) < min_d:
                    min_d = distance(e, k_arr[j])
                    ki = j
            cla_temp[ki].append(e)
            cla_head[ki].append(e[2])

        # 迭代更新聚类中心
        for k in range(k_arr.__len__()):
            if n - 1 == i:
                break
            k_arr[k] = means(cla_temp[k])
            cla_temp[k] = []
            cla_head[k] = []

    val = 0
    for cluster in cla_temp:
        prev_pos = None
        for item in cluster:
            pos = [item[0], item[1]]
            if prev_pos is not None:
                val += cheby(prev_pos, pos)
            prev_pos = pos
    return val, cla_temp


def genetic_based_placement_route_schedule(component_data, pcb_data, component_assign, feeder_slot_assign):
    # basic parameter
    # crossover rate & mutation rate: 80% & 10%
    # population size: 30
    # the number of generation: 100
    crossover_rate, mutation_rate = 0.6, 0.2
    population_size, n_generations = 50, 300

    with tqdm(total=n_generations) as pbar:
        pbar.set_description('ILP genetic process')

        # initial solution
        population, pop_val, pop_cluster = [], [], []
        for _ in range(population_size):
            pop_permutation = list_range(len(pcb_data))
            random.shuffle(pop_permutation)
            population.append(pop_permutation)

            # calculate fitness value
            val, cluster = cal_individual_cluster_val(pcb_data, pop_permutation, component_assign, feeder_slot_assign)
            pop_val.append(val)
            pop_cluster.append(cluster)

        best_pop_val = []
        for _ in range(n_generations):
            pop_index = np.argmin(pop_val)
            best_pop_val.append(min(pop_val))

            best_cluster = pop_cluster[pop_index]
            print('current best value', best_pop_val[-1])
            # min-max convert
            max_val = 1.5 * max(pop_val)
            sel_pop_val = list(map(lambda v: max_val - v, pop_val))

            # crossover and mutation
            new_population, new_pop_val, new_pop_cluster = [], [], []
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

                    val, cluster = cal_individual_cluster_val(pcb_data, offspring1, component_assign, feeder_slot_assign)
                    new_pop_val.append(val)
                    new_pop_cluster.append(cluster)

                    val, cluster = cal_individual_cluster_val(pcb_data, offspring2, component_assign, feeder_slot_assign)
                    new_pop_val.append(val)
                    new_pop_cluster.append(cluster)

            # selection
            top_k_index = get_top_k_value(pop_val, population_size - len(new_population), reverse=False)
            for index in top_k_index:
                new_population.append(population[index])
                new_pop_val.append(pop_val[index])
                new_pop_cluster.append(pop_cluster[index])

            population, pop_val, pop_cluster = new_population, new_pop_val, new_pop_cluster
            pbar.update(1)

        # print('best cluster: ', best_cluster)
        assigned_feeder = [45] * max_head_index
        placement_result, head_sequence_result = [], []
        for cluster in best_cluster:        # 四元组，分别为（X, Y, HD, IDX）
            cycle_placement = [-1 for _ in range(max_head_index)]
            for item in cluster:
                cycle_placement[item[2]] = item[3]

            placement_result.append(cycle_placement)
            head_sequence_result.append(dynamic_programming_cycle_path(pcb_data, cycle_placement, assigned_feeder))

    return placement_result, head_sequence_result


def route_saving_generation(component_data, pcb_data, component_assign, feeder_assign):
    placement_result, head_sequence_result = [], []
    mount_point_index = [[] for _ in range(len(component_data))]
    mount_point_pos = [[] for _ in range(len(component_data))]
    for i in range(len(pcb_data)):
        part = pcb_data.loc[i]['part']
        component_index = component_data[component_data['part'] == part].index.tolist()[0]
        # 记录贴装点序号索引和对应的位置坐标
        mount_point_index[component_index].append(i)
        mount_point_pos[component_index].append([pcb_data.loc[i]['x'], pcb_data.loc[i]['y']])

    # 假定初始位置
    deport = slotf1_pos
    deport[0] += 45 * slot_interval
    saving = []
    n = len(pcb_data)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # todo: 目前仅处理一种元件的情形
            dist = eular(deport, mount_point_pos[0][i]) + eular(deport, mount_point_pos[0][j])
            dist -= eular(mount_point_pos[0][i], mount_point_pos[0][j], 30)
            saving.append([i, j, dist])

    saving = sorted(saving, key=lambda x: x[2], reverse=True)

    routes = []
    for i in range(n):
        routes.append([i])

    for i in range(len(saving)):
        start_route, end_route = [], []
        for j in range(len(routes)):
            if saving[i][0] == routes[j][-1]:
                end_route = routes[j]
            elif saving[i][1] == routes[j][0]:
                start_route = routes[j]

            if len(end_route) > 0 and len(start_route) > 0:
                route_store = end_route + start_route

                if len(route_store) <= max_head_index:
                    routes.remove(start_route)
                    routes.remove(end_route)
                    routes.append(route_store)
                break
    for i in range(len(routes)):
        if mount_point_pos[0][routes[i][0]][0] > mount_point_pos[0][routes[i][-1]][0]:
            routes[i] = routes[i][::-1]

    val = 0
    for idx_route, route in enumerate(routes):
        prev_pos = None
        for head, i in enumerate(route):
            pos = mount_point_pos[0][i]
            pos[0] -= head * head_interval
            if prev_pos is not None:
                val += cheby(prev_pos, pos)
            prev_pos = pos

    index_mount = set()
    for route in routes:
        pos_x, pos_y = [], []
        for index, data in pcb_data.iterrows():
            if index in index_mount:
                continue
            pos_x.append(data['x'])
            pos_y.append(data['y'])

        mount_pos = []
        for head in range(len(route)):
            index = route[head]
            index_mount.add(index)

            x, y = pcb_data.iloc[index]['x'], pcb_data.iloc[index]['y']
            plt.text(x, y + 0.1, 'HD%d' % (head + 1), ha='center', va='bottom', size=10)
            plt.plot([x, x - head * head_interval], [y, y], linestyle='-.', color='black', linewidth=1)
            mount_pos.append([x - head * head_interval, y])
            plt.plot(mount_pos[-1][0], mount_pos[-1][1], marker='^', color='red', markerfacecolor='white')

        plt.scatter(pos_x, pos_y, s=8)
        # 绘制贴装路径
        for i in range(len(mount_pos) - 1):
            plt.plot([mount_pos[i][0], mount_pos[i + 1][0]], [mount_pos[i][1], mount_pos[i + 1][1]], color='blue',
                     linewidth=1)
        plt.show()

    return placement_result, head_sequence_result


def placement_route_schedule(component_data, pcb_data, component_assign, feeder_assign):
    placement_result, head_sequence_result = [], []

    return placement_result, head_sequence_result


def placement_time_estimate(component_data, pcb_data, nozzle_assign, component_assign, feeder_assign, placement_result,
                            head_sequence, hinter=True):
    # === 校验 ===
    total_points = 0
    for _, component_cycle in component_assign.iterrows():
        for head in range(max_head_index):
            if component_cycle['H' + str(head + 1)] == '':
                continue
            total_points += component_cycle['cycle']

    if total_points != len(pcb_data):
        warning_info = 'the number of placement points is not match with the PCB data. '
        warnings.warn(warning_info, UserWarning)
        return 0., (0, 0, 0)

    for placements in placement_result:
        for placement in placements:
            if placement == -1:
                continue
            total_points -= 1

    if total_points != 0:
        warnings.warn(
            'the optimization result of component assignment result and placement result are not consistent. ',
            UserWarning)
        return 0., (0, 0, 0)

    feeder_arrangement = defaultdict(set)
    for cycle_index, feeder_slots in feeder_assign.iterrows():
        for head in range(max_head_index):
            slot = feeder_slots['H' + str(head + 1)]
            if slot == '':
                continue
            part = component_assign.iloc[cycle_index]['H' + str(head + 1)]
            component_index = component_data[component_data['part'] == part].index.tolist()[0]
            feeder_arrangement[component_index].add(int(slot[1:]))

    for part, data in component_data.iterrows():
        if part in feeder_arrangement.keys() and data['feeder-limit'] < len(feeder_arrangement[part]):
            info = 'the number of arranged feeder of [' + data['part'] + '] exceeds the quantity limit'
            warnings.warn(info, UserWarning)
            return 0., (0, 0, 0)

    total_moving_time = .0                          # 总移动用时
    total_operation_time = .0                       # 操作用时
    total_nozzle_change_counter = 0                 # 总吸嘴更换次数
    total_pick_counter = 0                          # 总拾取次数
    total_mount_distance, total_pick_distance = .0, .0   # 贴装距离、拾取距离
    total_distance = 0                              # 总移动距离
    cur_pos, next_pos = anc_marker_pos, [0, 0]      # 贴装头当前位置

    # 初始化首个周期的吸嘴装配信息

    nozzle_assigned = [nozzle_assign.iloc[0]['H' + str(h + 1)] for h in range(max_head_index)]
    cycle = -1
    for cycle_set, component_cycle in component_assign.iterrows():
        for _ in range(int(component_cycle['cycle'])):
            cycle += 1
            pick_slot, mount_pos, mount_angle = [], [], []
            nozzle_pick_counter, nozzle_put_counter = 0, 0  # 吸嘴更换次数统计（拾取/放置分别算一次）
            for head in range(max_head_index):
                slot = feeder_assign.iloc[cycle_set]['H' + str(head + 1)]
                if slot != '':
                    pick_slot.append(int(slot[1:]) - head)

                if component_cycle['H' + str(head + 1)] == '':
                    continue
                nozzle = nozzle_assign.iloc[cycle_set]['H' + str(head + 1)]
                if nozzle != nozzle_assigned[head]:
                    if nozzle_assigned[head] != 'Empty':
                        nozzle_put_counter += 1
                    nozzle_pick_counter += 1
                    nozzle_assigned[head] = nozzle

            # ANC处进行吸嘴更换
            if nozzle_pick_counter + nozzle_put_counter > 0:
                next_pos = anc_marker_pos
                total_moving_time += max(axis_moving_time(cur_pos[0] - next_pos[0], 0),
                                         axis_moving_time(cur_pos[1] - next_pos[1], 1))
                total_distance += max(abs(cur_pos[0] - next_pos[0]), abs(cur_pos[1] - next_pos[1]))
                cur_pos = next_pos

            # 贴装路径
            for head in head_sequence[cycle]:
                index = placement_result[cycle][head]
                if index == -1:
                    continue
                mount_pos.append([pcb_data.loc[index]['x'] - head * head_interval + stopper_pos[0],
                                  pcb_data.loc[index]['y'] + stopper_pos[1]])
                mount_angle.append(pcb_data.loc[index]['r'])

            pick_slot = list(set(pick_slot))
            # 以下修改为适配 MIP 模型
            pick_slot = sorted(pick_slot, reverse=mount_pos[0][0] < mount_pos[-1][0])   # 拾取路径由贴装点相对位置确定

            # 拾取路径
            for slot in pick_slot:
                if slot < max_slot_index // 2:
                    next_pos = [slotf1_pos[0] + slot_interval * (slot - 1), slotf1_pos[1]]
                else:
                    next_pos = [slotr1_pos[0] - slot_interval * (max_slot_index - slot - 1), slotr1_pos[1]]
                total_operation_time += t_pick
                total_pick_counter += 1
                total_moving_time += max(axis_moving_time(cur_pos[0] - next_pos[0], 0),
                                         axis_moving_time(cur_pos[1] - next_pos[1], 1))
                total_distance += max(abs(cur_pos[0] - next_pos[0]), abs(cur_pos[1] - next_pos[1]))
                if slot != pick_slot[0]:
                    total_pick_distance += max(abs(cur_pos[0] - next_pos[0]), abs(cur_pos[1] - next_pos[1]))
                cur_pos = next_pos

            # todo: 省略了固定相机检测过程
            # 单独计算贴装路径
            for cntPoints in range(len(mount_pos) - 1):
                total_mount_distance += max(abs(mount_pos[cntPoints][0] - mount_pos[cntPoints + 1][0]),
                                            abs(mount_pos[cntPoints][1] - mount_pos[cntPoints + 1][1]))

            # 考虑R轴预旋转，补偿同轴角度转动带来的额外贴装用时
            total_operation_time += head_rotary_time(mount_angle[0])  # 补偿角度转动带来的额外贴装用时
            total_operation_time += t_nozzle_put * nozzle_put_counter + t_nozzle_pick * nozzle_pick_counter
            for pos in mount_pos:
                total_operation_time += t_place
                total_moving_time += max(axis_moving_time(cur_pos[0] - pos[0], 0),
                                         axis_moving_time(cur_pos[1] - pos[1], 1))
                total_distance += max(abs(cur_pos[0] - pos[0]), abs(cur_pos[1] - pos[1]))
                cur_pos = pos

            total_nozzle_change_counter += nozzle_put_counter + nozzle_pick_counter

    total_time = total_moving_time + total_operation_time
    minutes, seconds = int(total_time // 60), int(total_time) % 60
    millisecond = int((total_time - minutes * 60 - seconds) * 60)

    if hinter:
        total_cycle = 0
        for _, component_cycle in component_assign.iterrows():
            total_cycle += component_cycle['cycle']

        print('-Cycle counter: {}'.format(total_cycle))
        print('-Nozzle change counter: {}'.format(total_nozzle_change_counter // 2))
        print('-Pick operation counter: {}'.format(total_pick_counter))

        print('-Expected mounting tour length: {} mm'.format(total_mount_distance))
        print('-Expected picking tour length: {} mm'.format(total_pick_distance))
        print('-Expected total tour length: {} mm'.format(total_distance))

        print('-Expected total moving time: {} s'.format(total_moving_time))
        print('-Expected total operation time: {} s'.format(total_operation_time))

        if minutes > 0:
            print('-Mounting time estimation:  {:d} min {} s {:2d} ms ({:.3f}s)'.format(minutes, seconds, millisecond,
                                                                                        total_time))
        else:
            print('-Mounting time estimation:  {} s {:2d} ms ({:.3f}s)'.format(seconds, millisecond, total_time))

    return total_time


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
    pcb_data, component_data, _ = load_data(params.filename, feeder_limit=1, auto_register=params.auto_register)

    _, nozzle_assign, component_assign, feeder_assign = gurobi_optimizer(pcb_data, component_data, feeder_data=None,
                                                                         initial=True, hinter=True)

    placement_assign, head_sequence_assign = genetic_based_placement_route_schedule(component_data, pcb_data,
                                                                                    component_assign, feeder_assign)

    # placement_assign, head_sequence_assign = greedy_placement_route_generation(component_data, pcb_data,
    #                                                                            component_assign, feeder_assign)

    # placement_assign, head_sequence_assign = placement_route_schedule(component_data, pcb_data, component_assign,
    #                                                                   feeder_assign)

    placement_time_estimate(component_data, pcb_data, nozzle_assign, component_assign, feeder_assign, placement_assign,
                            head_sequence_assign)

    index_mount = set()
    cycle_show = 0
    for cycle_index, placement_points in enumerate(placement_assign):
        pos_x, pos_y = [], []
        for index, data in pcb_data.iterrows():
            if index in index_mount:
                continue
            pos_x.append(data['x'])
            pos_y.append(data['y'])

        mount_pos = []
        for head in head_sequence_assign[cycle_index]:
            index = placement_assign[cycle_index][head]
            index_mount.add(index)
            if cycle_index < cycle_show:
                continue
            x, y = pcb_data.iloc[index]['x'], pcb_data.iloc[index]['y']
            plt.text(x, y + 0.1, 'HD%d' % (head + 1), ha='center', va='bottom', size=10)
            # plt.plot([x, x - head * head_interval], [y, y], linestyle='-.', color='black', linewidth=1)
            mount_pos.append([x - head * head_interval, y])
            plt.plot(mount_pos[-1][0], mount_pos[-1][1], marker='^', color='red', markerfacecolor='white')

        if cycle_index < cycle_show:
            continue
        plt.scatter(pos_x, pos_y, s=8)
        # 绘制贴装路径
        for i in range(len(mount_pos) - 1):
            plt.plot([mount_pos[i][0], mount_pos[i + 1][0]], [mount_pos[i][1], mount_pos[i + 1][1]], color='blue',
                     linewidth=1)
        plt.show()

    # genetic_algorithm(pcb_data, component_data)


if __name__ == '__main__':
    main()
