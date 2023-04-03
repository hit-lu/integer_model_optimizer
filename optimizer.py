import copy
import math
import matplotlib.pyplot as plt
import pandas as pd
import time
import argparse

from ortools.sat.python import cp_model
from collections import defaultdict
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


@timer_wrapper
def ortools_main():
    # warnings.simplefilter('ignore')
    # 参数解析
    parser = argparse.ArgumentParser(description='assembly line optimizer implementation')
    parser.add_argument('--filename', default='PCB.txt', type=str, help='load pcb data')
    parser.add_argument('--auto_register', default=1, type=int, help='register the component according the pcb data')
    params = parser.parse_args()

    # 结果输出显示所有行和列
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # 加载PCB数据
    pcb_data, component_data, _ = load_data(params.filename, default_feeder_limit=1,
                                            cp_auto_register=params.auto_register)  # 加载PCB数据

    # data preparation: convert data to index
    component_list, nozzle_list = defaultdict(int), defaultdict(int)
    cpidx_2_part, nzidx_2_nozzle = {}, {}
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

    weight_cycle, weight_nz_change, weight_pick = 1, 6, 2

    r = 1
    I, J = len(cpidx_2_part.keys()), len(nzidx_2_nozzle.keys())
    L = 3
    H = 6       # max head num
    S = r * I   # the available feeder num
    M = 10000    # a sufficient large number
    HC = [[0 for _ in range(J)] for _ in range(I)]
    for i in range(I):
        for index, part in cpidx_2_part.items():
            cp_idx = component_data[component_data['part'] == part].index.tolist()[0]
            nozzle = component_data.loc[cp_idx]['nz']

            for j in range(J):
                HC[index][j] = 1 if nzidx_2_nozzle[j] == nozzle else 0

    # === phase 1: mathematical model solver ===
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()

    # === Decision Variables ===
    # the number of components of type i that are placed by nozzle type j on placement head k
    x = {}
    for i in range(I):
        for s in range(S + r * (H - 1)):
            for h in range(H):
                for l in range(L):
                    x[i, s, h, l] = model.NewBoolVar('x_{}_{}_{}_{}'.format(i, s, h, l))

    c = {}
    for i in range(I):
        for s in range(S):
            for h in range(H):
                for l in range(L):
                    c[i, s, h, l] = model.NewIntVar(0, len(pcb_data) // H + 1, 'q_{}_{}_{}_{}'.format(i, s, h, l))

    w = {}
    for l in range(L):
        w[l] = model.NewIntVar(0, len(pcb_data) // H + 1, 'w_{}_'.format(l))

    p = {}
    for s in range(-(H - 1) * r, S):
        for l in range(L):
            p[s, l] = model.NewBoolVar('p_{}_{}'.format(s, l))

    PU = {}
    for s in range(-(H - 1) * r, S):
        for l in range(L):
            PU[s, l] = model.NewIntVar(0, M, 'PU_{}_{}'.format(s, l))

    WL = model.NewIntVar(0, len(pcb_data) // H + 1, 'WL')

    z = {}
    for j in range(J):
        for h in range(H):
            for l in range(L):
                z[j, h, l] = model.NewBoolVar('z_{}_{}_{}'.format(j, h, l))

    # the total number of nozzle changes on placement head h
    NC = {}
    for h in range(H):
        NC[h] = model.NewIntVar(0, J, 'N_{}'.format(h))

    # d[l, h] := 2 if a change of nozzles in the level l + 1 on placement head h
    # d[l, h] := 1 if there are no batches placed on levels higher than l
    d = {}
    for l in range(L - 1):
        for h in range(H):
            d[l, h] = model.NewBoolVar('d_{}_{}'.format(l, h))

    d_abs = {}
    for l in range(L - 1):
        for j in range(J):
            for h in range(H):
                d_abs[l, j, h] = model.NewBoolVar('d_abs_{}_{}_{}'.format(l, j, h))

    f = {}
    for i in range(I):
        for s in range(S):
            f[s, i] = model.NewBoolVar('f_{}_{}'.format(s, i))

    # == 优化目标 ===
    model.Minimize(weight_cycle * WL + weight_nz_change * sum(NC[h] for h in range(H)) + weight_pick * sum(
        PU[s, l] for s in range(-(H - 1) * r, S) for l in range(L)))

    # === Constraint ===
    # 1. 工作完整性
    for i in range(I):
        for s in range(S):
            for h in range(H):
                for l in range(L):
                    model.Add(c[i, s, h, l] <= M * x[i, s, h, l])
                    model.Add(c[i, s, h, l] <= w[l])
                    model.Add(c[i, s, h, l] >= w[l] - M * (1 - x[i, s, h, l]))

    for i in range(I):
        part = cpidx_2_part[i]
        model.Add(sum(c[i, s, h, l] for s in range(S) for h in range(H) for l in range(L)) == component_list[part])

    # 2. 同时拾取
    for s in range(-(H - 1) * r, S):
        for l in range(L):
            rng = list(range(max(0, -math.floor(s / r)), H))
            model.Add(sum(x[i, s + h * r, h, l] for h in rng for i in range(I)) <= M * p[s, l])
            model.Add(sum(x[i, s + h * r, h, l] for h in rng for i in range(I)) >= p[s, l])

    for s in range(-(H - 1) * r, S):
        for l in range(L):
            model.Add(PU[s, l] <= M * p[s, l])
            model.Add(PU[s, l] <= w[l])
            model.Add(PU[s, l] >= w[l] - M * (1 - p[s, l]))

    # 3. 工作周期
    model.Add(sum(w[l] for l in range(L)) == WL)

    for l in range(L - 1):
        model.Add(w[l] >= w[l + 1])
        model.Add(sum(x[i, s, h, l] for i in range(I) for s in range(S) for h in range(H)) >= sum(
            x[i, s, h, l + 1] for i in range(I) for s in range(S) for h in range(H)))

    # 4. 吸嘴更换
    for l in range(L - 1):
        for j in range(J):
            for h in range(H):
                # model.AddAbsEquality(target, var) <=> target == |var|
                model.AddAbsEquality(d_abs[l, j, h], z[j, h, l] - z[j, h, l + 1])

    for h in range(H):
        for l in range(L - 1):
            model.Add(2 * d[l, h] == sum(d_abs[l, j, h] for j in range(J)))

    for h in range(H):
        model.Add(NC[h] == sum(d[l, h] for l in range(L - 1)))

    # 5. 吸嘴-元件类型一致性
    for i in range(I):
        for h in range(H):
            for l in range(L):
                model.Add(sum(x[i, s, h, l] for s in range(S)) <= sum(HC[i][j] * z[j, h, l] for j in range(J)))

    # 6. 其它
    for h in range(H):
        for l in range(L):
            model.Add(sum(x[i, s, h, l] for s in range(S) for i in range(I)) <= 1)
            model.Add(sum(z[j, h, l] for j in range(J)) <= 1)

    for i in range(I):
        for s in range(S):
            model.Add(sum(x[i, s, h, l] for h in range(H) for l in range(L)) >= f[s, i])
            model.Add(sum(x[i, s, h, l] for h in range(H) for l in range(L)) <= M * f[s, i])

    # 任意一个元件最多放一个槽位
    for i in range(I):
        model.Add(sum(f[s, i] for s in range(S)) <= 1)

    # 任意一个槽位最多放一个元件
    for s in range(S):
        model.Add(sum(f[s, i] for i in range(I)) <= 1)

    solution_callback = SolutionCallback()
    solver.parameters.max_time_in_seconds = 5
    solver.parameters.mip_max_bound = 40
    status = solver.Solve(model, solution_callback)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        res = 'Optimal solution' if status == cp_model.OPTIMAL else 'Feasible solution'
        print(res)
        print('total cost = {}'.format(solver.ObjectiveValue()))
        print('cycle = {}, nozzle change = {}, pick up = {}'.format(sum(solver.Value(w[l]) for l in range(L)),
                                                                    sum(solver.Value(NC[h]) for h in range(H)), sum(
                solver.Value(PU[s, l]) for s in range(-(H - 1) * r, S) for l in range(L))))

        print('workload: ')
        for l in range(L):
            print(solver.Value(w[l]), end=', ')

        print('')
        print('pick up information')
        for l in range(L):
            print('level ' + str(l + 1), ': ', end='')
            for s in range(-(H - 1) * r, S):
                print(solver.Value(PU[s, l]), end=', ')
            print('')

        print('')
        print('result')
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        columns = ['H{}'.format(i + 1) for i in range(H)] + ['cycle']

        nozzle_assign = pd.DataFrame(columns=columns)
        component_assign = pd.DataFrame(columns=columns)
        feeder_assign = pd.DataFrame(columns=columns)

        for l in range(L):
            if solver.Value(w[l]) == 0:
                break

            nozzle_assign.loc[l, 'cycle'] = solver.Value(w[l])
            component_assign.loc[l, 'cycle'] = solver.Value(w[l])
            feeder_assign.loc[l, 'cycle'] = solver.Value(w[l])

            for h in range(H):
                bAssign = False
                for i in range(I):
                    for s in range(S):
                        if solver.Value(x[i, s, h, l]) == 1:
                            bAssign = True
                            component_assign.loc[l, 'H{}'.format(h + 1)] = 'CP' + str(i) + ': ' + cpidx_2_part[i]
                            feeder_assign.loc[l, 'H{}'.format(h + 1)] = 'F' + str(s + 1)

                for j in range(J):
                    if solver.Value(z[j, h, l]) == 1:
                        nozzle_assign.loc[l, 'H{}'.format(h + 1)] = nzidx_2_nozzle[j]

                if not bAssign:
                    nozzle_assign.loc[l, 'H{}'.format(h + 1)] = 'A'
                    component_assign.loc[l, 'H{}'.format(h + 1)] = ''
                    feeder_assign.loc[l, 'H{}'.format(h + 1)] = ''

        print(nozzle_assign)
        print(component_assign)
        print(feeder_assign)
    else:
        print('no solution found')


def list_range(start, end=None):
    return list(range(start)) if end is None else list(range(start, end))


@timer_wrapper
def gurobi_main():
    parser = argparse.ArgumentParser(description='assembly line optimizer implementation')
    parser.add_argument('--filename', default='PCB.txt', type=str, help='load pcb data')
    parser.add_argument('--auto_register', default=1, type=int, help='register the component according the pcb data')
    params = parser.parse_args()

    # 结果输出显示所有行和列
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # 加载PCB数据
    pcb_data, component_data, _ = load_data(params.filename, default_feeder_limit=1,
                                            cp_auto_register=params.auto_register)  # 加载PCB数据

    # data preparation: convert data to index
    component_list, nozzle_list = defaultdict(int), defaultdict(int)
    cpidx_2_part, nzidx_2_nozzle = {}, {}
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

    weight_cycle, weight_nz_change, weight_pick = 2, 6, 1

    r = 1
    I, J = len(cpidx_2_part.keys()), len(nzidx_2_nozzle.keys())
    L = 2       # todo: how to deal with this hyper-parameter
    H = 6       # max head num
    S = r * I   # the available feeder num
    M = 1000    # a sufficient large number
    HC = [[0 for _ in range(J)] for _ in range(I)]
    for i in range(I):
        for index, part in cpidx_2_part.items():
            cp_idx = component_data[component_data['part'] == part].index.tolist()[0]
            nozzle = component_data.loc[cp_idx]['nz']

            for j in range(J):
                HC[index][j] = 1 if nzidx_2_nozzle[j] == nozzle else 0

    mdl = Model('SMT')
    x = mdl.addVars(list_range(I), list_range(S + r * (H - 1)), list_range(H), list_range(L), vtype=GRB.BINARY)
    c = mdl.addVars(list_range(I), list_range(S), list_range(H), list_range(L), vtype=GRB.INTEGER)
    p = mdl.addVars(list_range(-(H - 1) * r, S), list_range(L), vtype=GRB.BINARY)
    z = mdl.addVars(list_range(J), list_range(H), list_range(L), vtype=GRB.BINARY)
    d = mdl.addVars(list_range(L - 1), list_range(H), vtype=GRB.BINARY)
    d_abs = mdl.addVars(list_range(L - 1), list_range(J), list_range(H), vtype=GRB.BINARY)
    f = mdl.addVars(list_range(S), list_range(I), vtype=GRB.BINARY)

    PU = mdl.addVars(list_range(-(H - 1) * r, S), list_range(L), vtype=GRB.INTEGER)
    WL = mdl.addVars(list_range(L), vtype=GRB.INTEGER)
    NC = mdl.addVars(list_range(H), vtype=GRB.INTEGER)

    mdl.modelSense = GRB.MINIMIZE  # 目标为最小化
    mdl.setObjective(weight_cycle * quicksum(WL[l] for l in range(L)) + weight_nz_change * quicksum(
        NC[h] for h in range(H)) + weight_pick * quicksum(PU[s, l] for s in range(-(H - 1) * r, S) for l in range(L)))

    # 工作完整性
    mdl.addConstrs(
        c[i, s, h, l] <= M * x[i, s, h, l] for i in range(I) for s in range(S) for h in range(H) for l in range(L))
    mdl.addConstrs(c[i, s, h, l] <= WL[l] for i in range(I) for s in range(S) for h in range(H) for l in range(L))
    mdl.addConstrs(
        c[i, s, h, l] >= WL[l] - M * (1 - x[i, s, h, l]) for i in range(I) for s in range(S) for h in range(H) for l in
        range(L))

    mdl.addConstrs(
        quicksum(c[i, s, h, l] for s in range(S) for h in range(H) for l in range(L)) == component_list[cpidx_2_part[i]]
        for i in range(I))

    # 同时拾取
    for s in range(-(H - 1) * r, S):
        for l in range(L):
            rng = list(range(max(0, -math.floor(s / r)), H))
            mdl.addConstr(quicksum(x[i, s + h * r, h, l] for h in rng for i in range(I)) <= M * p[s, l], name='')
            mdl.addConstr(quicksum(x[i, s + h * r, h, l] for h in rng for i in range(I)) >= p[s, l], name='')

    mdl.addConstrs(PU[s, l] <= M * p[s, l] for s in range(-(H - 1) * r, S) for l in range(L))
    mdl.addConstrs(PU[s, l] <= WL[l] for s in range(-(H - 1) * r, S) for l in range(L))
    mdl.addConstrs(PU[s, l] >= WL[l] - M * (1 - p[s, l]) for s in range(-(H - 1) * r, S) for l in range(L))

    # 工作周期
    mdl.addConstrs(WL[l] >= WL[l + 1] for l in range(L - 1))

    # 吸嘴更换
    mdl.addConstrs(
        d_abs[l, j, h] >= z[j, h, l] - z[j, h, l + 1] for l in range(L - 1) for j in range(J) for h in range(H))
    mdl.addConstrs(
        d_abs[l, j, h] >= -z[j, h, l] - z[j, h, l + 1] for l in range(L - 1) for j in range(J) for h in range(H))

    mdl.addConstrs(
        2 * d[l, h] == quicksum(d_abs[l, j, h] for j in range(J)) for l in range(L - 1) for h in range(H))

    mdl.addConstrs(NC[h] == quicksum(d[l, h] for l in range(L - 1)) for h in range(H))

    # 吸嘴-元件类型一致性
    mdl.addConstrs(
        quicksum(x[i, s, h, l] for s in range(S)) <= sum(HC[i][j] * z[j, h, l] for j in range(J)) for i in range(I) for
        h in range(H) for l in range(L))

    # 其它
    mdl.addConstrs(quicksum(x[i, s, h, l] for s in range(S) for i in range(I)) <= 1 for h in range(H) for l in range(L))
    mdl.addConstrs(quicksum(z[j, h, l] for j in range(J)) <= 1 for h in range(H) for l in range(L))
    mdl.addConstrs(
        quicksum(x[i, s, h, l] for h in range(H) for l in range(L)) >= f[s, i] for i in range(I) for s in range(S))

    mdl.addConstrs(
        quicksum(x[i, s, h, l] for h in range(H) for l in range(L)) <= M * f[s, i] for i in range(I) for s in range(S))

    mdl.addConstrs(quicksum(f[s, i] for s in range(S)) <= 1 for i in range(I))
    mdl.addConstrs(quicksum(f[s, i] for i in range(I)) <= 1 for s in range(S))

    # 变量取值范围约束
    mdl.addConstrs(c[i, s, h, l] >= 0 for i in range(I) for s in range(S) for h in range(H) for l in range(L))
    mdl.addConstrs(
        c[i, s, h, l] <= len(pcb_data) // H + 1 for i in range(I) for s in range(S) for h in range(H) for l in range(L))

    mdl.addConstrs(WL[l] >= 0 for l in range(L))
    mdl.addConstrs(WL[l] <= len(pcb_data) // H + 1 for l in range(L))

    mdl.addConstrs(PU[s, l] <= M for s in range(-(H - 1) * r, S) for l in range(L))
    mdl.addConstrs(PU[s, l] >= 0 for s in range(-(H - 1) * r, S) for l in range(L))

    mdl.addConstrs(NC[h] >= 0 for h in range(H))
    mdl.addConstrs(NC[h] <= J for h in range(H))

    mdl.optimize()
    print('total cost = {}'.format(mdl.objval))
    print('cycle = {}, nozzle change = {}, pick up = {}'.format(quicksum(WL[l].x for l in range(L)),
          quicksum(NC[h].x for h in range(H)), quicksum(PU[s, l].x for s in range(-(H - 1) * r, S) for l in range(L))))

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
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    columns = ['H{}'.format(i + 1) for i in range(H)] + ['cycle']

    nozzle_assign = pd.DataFrame(columns=columns)
    component_assign = pd.DataFrame(columns=columns)
    feeder_assign = pd.DataFrame(columns=columns)

    for l in range(L):
        if WL[l].x == 0:
            break

        nozzle_assign.loc[l, 'cycle'] = WL[l].x
        component_assign.loc[l, 'cycle'] = WL[l].x
        feeder_assign.loc[l, 'cycle'] = WL[l].x

        for h in range(H):
            bAssign = False
            for i in range(I):
                for s in range(S):
                    if abs(x[i, s, h, l].x - 1) < 1e-10:
                        bAssign = True
                        component_assign.loc[l, 'H{}'.format(h + 1)] = 'CP' + str(i) + ': ' + cpidx_2_part[i]
                        feeder_assign.loc[l, 'H{}'.format(h + 1)] = 'F' + str(s + 1)

            for j in range(J):
                if abs(z[j, h, l].x - 1) < 1e-10:
                    nozzle_assign.loc[l, 'H{}'.format(h + 1)] = nzidx_2_nozzle[j]

            if not bAssign:
                nozzle_assign.loc[l, 'H{}'.format(h + 1)] = 'A'
                component_assign.loc[l, 'H{}'.format(h + 1)] = ''
                feeder_assign.loc[l, 'H{}'.format(h + 1)] = ''

    print(nozzle_assign)
    print(component_assign)
    print(feeder_assign)


if __name__ == '__main__':
    gurobi_main()


