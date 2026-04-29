import random
import os
import gurobipy
from gurobipy import *
import argparse
import numpy as np
import pandas as pd
import re
import math
import sys
# import generate_data_files as gdf
import time
from revise_data import *

start_time = time.time()

np.random.seed(0)
""" The input parameter of stage 1"""

# retailer_number, product_kind, vehicle_number, Ui, cij, pim, Qk, bk, vim, him, Iim0, dim_real, dim \
#     = parse_data_file(sys.argv[1])
# retailer_number, product_kind, vehicle_number, Ui, cij, pim, Qk, bk, vim, him, Iim0, dim_real , dim\
#     = parse_data_file('example_data_prod_more/mirplr-10-50-5-5.dat')
data_dic = parse_data_file(sys.argv[1])
# data_dic = parse_data_file('./rho_sensitivity/mirplr-5-9-5-1.dat')

retailer_number, product_kind, vehicle_number, Ui, cij, pim, Qk, bk, vim, him, Iim0, dim_real, dim, node_product_relationships, product_node_relationships \
    = data_dic["num_customers"], data_dic["num_products"], data_dic["num_vehicles"], data_dic["Ui"], data_dic["cij"], \
    data_dic["pim"], \
    data_dic["Qk"], data_dic["bk"], data_dic["vim"], data_dic["him"], data_dic["Iim0"], data_dic["dim_actual"], \
    data_dic["dim_predict"], data_dic["node_product_relationships"], data_dic["product_node_relationships"]
nodes_number = retailer_number + 1
""" The input parameter of stage 2"""
coefficient = 1.2
cij_trans = [[element * coefficient for element in row] for row in cij]
# print("retailer_number:", retailer_number)
# print("product_kind:", product_kind)
# print("vehicle_number:", vehicle_number)
# print("Ui:", Ui)
# print("vim:", pim)
# print("Qk:", Qk)
# print("bk:", bk)
# print("Oim:", vim)
# print("him:", him)
# print("Iim0:", Iim0)
# print("dim_real:", dim_real)
# print('dim:',dim)
'''controlled parameters'''
tau_coef = 1
tau_T = retailer_number * product_kind * tau_coef
# DSR0 = 0.9
bigM = 999999
smallS = 0.00001
# gamma = 1
# parser = argparse.ArgumentParser()
# # 添加参数
# parser.add_argument("data_file", type=str, help="Path to the data file (.dat)")
# parser.add_argument("--b_lb", type=float, required=True, help="Lower bound for b")
# parser.add_argument("--b_ub", type=float, required=True, help="Upper bound for b")
# parser.add_argument("--b_total", type=float, required=True, help="Total value of b")
# parser.add_argument("--rho", type=float, required=True, help="Rho value")
#
# # 解析命令行参数
# args = parser.parse_args()
#
#
#
# b_lb=args.b_lb
# b_ub=args.b_ub
# b_total =args.b_total
# rho = args.rho
#
#
# b = b_total
# b_comp=random.uniform(b_lb,b_ub)
# b_sync=random.uniform(b_lb,b_ub)
# b_conflict=random.uniform(b_lb,b_ub)
# b_region=random.uniform(b_lb,b_ub)
#
# # Define uncertainty budgets (example values)
# Gamma = b * retailer_number * product_kind  # Global budget
# Gamma_comp, Gamma_sync, Gamma_conflict, Gamma_region = parse_relationships_and_define_gamma(node_product_relationships,
#                                                                                             product_node_relationships,
#                                                                                             b_comp=b_comp,
#                                                                                             b_sync=b_sync,
#                                                                                             b_conflict=b_conflict,
#                                                                                             b_region=b_region)

b=0.3
rho=0.5
Gamma = b * retailer_number * product_kind  # Global budget
Gamma_comp, Gamma_sync, Gamma_conflict, Gamma_region = parse_relationships_and_define_gamma(node_product_relationships,
                                                                                            product_node_relationships)

""" build initial master problem """
""" Create variables """
master = Model('master problem')

# print(retailer_number, product_kind, vehicle_number, Ui, x_co, y_co, dim_real, pim, Qk, bk, vim, him, Iim0)

master.setParam('Outputflag', 0)
rm = {}
xijk = {}
yik = {}
Iim = {}
qimk = {}
sdim = {}
# sdim
for i in range(retailer_number):
    for m in range(product_kind):
        sdim[i, m] = master.addVar(lb=0, ub=dim[i][m], vtype=GRB.CONTINUOUS, name=f'sdim_{i + 1}_{m + 1}')
# rm
for m in range(product_kind):
    rm[m] = master.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'v_{m + 1}')
# xijk
for i in range(nodes_number):
    xijk[i] = {}
    for j in range(nodes_number):
        if i < j:
            xijk[i][j] = {}
            for k in range(vehicle_number):
                if i == 0:
                    xijk[i][j][k] = master.addVar(lb=0, ub=2, vtype=GRB.INTEGER, name=f'x_{i}_{j}_{k + 1}')
                else:
                    xijk[i][j][k] = master.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}_{k + 1}')
# yik
for i in range(nodes_number):
    yik[i] = {}
    for k in range(vehicle_number):
        yik[i][k] = master.addVar(vtype=GRB.BINARY, name=f'y_{i}_{k + 1}')
# Iim
for i in range(nodes_number):
    Iim[i] = {}
    for m in range(product_kind):
        Iim[i][m] = master.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'I_{i}_{m + 1}')
# qimk
for i in range(retailer_number):
    qimk[i] = {}
    for m in range(product_kind):
        qimk[i][m] = {}
        for k in range(vehicle_number):
            qimk[i][m][k] = master.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'q_{i + 1}_{m + 1}_{k + 1}')
# eta
eta = master.addVar(vtype=GRB.CONTINUOUS, name='eta')

# 参数算法中的参数设置
iter_cnt = 0
alpha = {}
beta = {}
alpha[iter_cnt] = 1
beta[iter_cnt] = 1

""" Set objective """
master_obj = LinExpr()
for i in range(retailer_number):
    for m in range(product_kind):
        for k in range(vehicle_number):
            master_obj.addTerms(pim[i][m], qimk[i][m][k])
for i in range(nodes_number):
    for j in range(nodes_number):
        if i < j:
            for k in range(vehicle_number):
                master_obj.addTerms(-alpha[iter_cnt] * cij[i][j], xijk[i][j][k])
for i in range(nodes_number):
    for m in range(product_kind):
        master_obj.addTerms(-alpha[iter_cnt] * him[i][m], Iim[i][m])
for k in range(vehicle_number):
    master_obj.addTerms(-alpha[iter_cnt] * bk[k], yik[0][k])
master_obj.addTerms(1, eta)
master.setObjective(master_obj, GRB.MAXIMIZE)

""" Add Constraints  """
# cons 2
for m in range(product_kind):
    qimk_ik_sum = LinExpr()
    for k in range(vehicle_number):
        for i in range(retailer_number):
            qimk_ik_sum.addTerms(1, qimk[i][m][k])
    master.addConstr(Iim[0][m] == Iim0[0][m] + rm[m] - qimk_ik_sum, name=f'cons(2)-m{m + 1}')
    # master.addConstr(Iim[0][m] - (Iim0[0][m] + rm[m] - qimk_ik_sum) <= smallS, name=f'cons(2)-m{m + 1}_1')
    # master.addConstr(Iim[0][m] - (Iim0[0][m] + rm[m] - qimk_ik_sum) <= -smallS, name=f'cons(2)-m{m + 1}_2')

# cons 3
for i in range(retailer_number):
    for m in range(product_kind):
        qimk_k_sum = LinExpr()
        # master.addConstr(sdim[i, m] >= 1 * dim[i][m])
        for k in range(vehicle_number):
            qimk_k_sum.addTerms(1, qimk[i][m][k])
        master.addConstr(Iim[i + 1][m] == Iim0[i + 1][m] + qimk_k_sum - sdim[i, m],
                         name=f'cons(3)-i{i + 1}-m{m + 1}')
        # master.addConstr(Iim[i + 1][m] - (Iim0[i + 1][m] + qimk_k_sum - sdim[i, m]) <= smallS,
        #                  name=f'cons(3)-i{i + 1}-m{m + 1}_1')
        # master.addConstr(Iim[i + 1][m] - (Iim0[i + 1][m] + qimk_k_sum - sdim[i, m]) >= -smallS,
        #                  name=f'cons(3)-i{i + 1}-m{m + 1}_2')

# cons 4
rm_m_sum = LinExpr()
I0m0_m_sum = 0
for m in range(product_kind):
    rm_m_sum.addTerms(vim[0][m], rm[m])
    I0m0_m_sum += Iim0[0][m]
master.addConstr(rm_m_sum + I0m0_m_sum * vim[0][m] <= Ui[0], name='cons(4)')
# cons 5
for i in range(retailer_number):
    qimk_km_sum = LinExpr()
    Iim0_m_sum = 0
    for m in range(product_kind):
        Iim0_m_sum += Iim0[i + 1][m]
        # yik_k_sum=LinExpr()
        for k in range(vehicle_number):
            qimk_km_sum.addTerms(vim[i][m], qimk[i][m][k])
            # yik_k_sum.addTerms(1,yik[i+1][k])
    master.addConstr(qimk_km_sum + Iim0_m_sum * vim[i][m] <= Ui[i + 1], name=f'cons(5)-i{i + 1}')
# cons 6
for i in range(retailer_number):
    for k in range(vehicle_number):
        qimk_m_sum = LinExpr()
        for m in range(product_kind):
            qimk_m_sum.addTerms(vim[i][m], qimk[i][m][k])
        master.addConstr(qimk_m_sum <= Ui[i + 1] * yik[i + 1][k], name=f'cons(6)-i{i + 1}-k{k + 1}')

# for i in range(retailer_number):
#     yik_k_sum = LinExpr()
#     for k in range(vehicle_number):
#         yik_k_sum.addTerms(1,yik[i+1][k])
#     master.addConstr(yik_k_sum<=1)
# cons 7
for k in range(vehicle_number):
    qimk_im_sum = LinExpr()
    for i in range(retailer_number):
        for m in range(product_kind):
            qimk_im_sum.addTerms(vim[i][m], qimk[i][m][k])
    master.addConstr(qimk_im_sum <= Qk[k] * yik[0][k], name=f'cons(7)-k{k + 1}')
# cons 8
for i in range(nodes_number):
    for k in range(vehicle_number):
        xijk_j_sum = LinExpr()
        xjik_j_sum = LinExpr()
        for j in range(nodes_number):
            if i < j:
                xijk_j_sum.addTerms(1, xijk[i][j][k])
            if j < i:
                xjik_j_sum.addTerms(1, xijk[j][i][k])
        master.addConstr(xijk_j_sum + xjik_j_sum == 2 * yik[i][k], name=f'cons(8)-i{i}-k{k + 1}')


# function 找出节点集合的全部真子集
def PowerSetsBinary(node_list):
    N = len(node_list)
    Ss = []
    for i in range(2 ** N):  # 子集个数，每循环一次一个子集
        Ss.append([])
        for j in range(N):  # 用来判断二进制下标为j的位置数是否为1
            if (i >> j) % 2:
                Ss[i].append(node_list[j])
    return Ss


# cons 9
Ss = PowerSetsBinary(range(retailer_number))
for S in Ss:
    if S != []:
        for g in S:
            for k in range(vehicle_number):
                xijk_ij_sum = LinExpr()
                yik_i_sum = LinExpr()
                for i in S:
                    yik_i_sum.addTerms(1, yik[i + 1][k])
                    for j in S:
                        if i < j:
                            xijk_ij_sum.addTerms(1, xijk[i + 1][j + 1][k])
                master.addConstr(xijk_ij_sum <= yik_i_sum - yik[g + 1][k], name=f'cons(9)-k{k + 1}-g{g + 1}')

# create new constraints
master.addConstr(eta <= bigM)
""" solve the model and output  """
master.Params.TimeLimit = 600
master.Params.NodefileStart = 0.5
master.optimize()

iter_a0 = 0
delta_0 = 0.01
iter_max_0 = 50
while (((master.ObjVal - eta.x) > delta_0) and (iter_a0 < iter_max_0)) or (iter_a0 == 0):
    master_numerator = 0
    master_denominator = 0
    for i in range(retailer_number):
        for m in range(product_kind):
            for k in range(vehicle_number):
                master_numerator += pim[i][m] * qimk[i][m][k].x
    for i in range(nodes_number):
        for j in range(nodes_number):
            if i < j:
                for k in range(vehicle_number):
                    master_denominator += cij[i][j] * xijk[i][j][k].x
    for i in range(nodes_number):
        for m in range(product_kind):
            master_denominator += him[i][m] * Iim[i][m].x
    for k in range(vehicle_number):
        master_denominator += bk[k] * yik[0][k].x
    if master_denominator == 0:
        alpha[iter_cnt] = 0
    else:
        alpha[iter_cnt] = master_numerator / master_denominator
    iter_a0 += 1

    """ reSet master objective """
    master_obj = LinExpr()
    for i in range(retailer_number):
        for m in range(product_kind):
            for k in range(vehicle_number):
                master_obj.addTerms(pim[i][m], qimk[i][m][k])
    for i in range(nodes_number):
        for j in range(nodes_number):
            if i < j:
                for k in range(vehicle_number):
                    master_obj.addTerms(-alpha[iter_cnt] * cij[i][j], xijk[i][j][k])
    for i in range(nodes_number):
        for m in range(product_kind):
            master_obj.addTerms(-alpha[iter_cnt] * him[i][m], Iim[i][m])
    for k in range(vehicle_number):
        master_obj.addTerms(-alpha[iter_cnt] * bk[k], yik[0][k])
    master_obj.addTerms(1, eta)
    master.setObjective(master_obj, GRB.MAXIMIZE)
    # solve the resulted master problem
    master.optimize()
    # if master.status != 2:
    #     # print('master(3 unbound):{}'.format(master.Status))
    #     master.computeIIS()
    #     master.write("master_0421.ilp")

# master.write('master.lp')
# 如果模型不可行，计算 IIS
# if master.Status == GRB.INFEASIBLE or master.Status == GRB.INF_OR_UNBD:
#     print("master Model is infeasible.")
#     # 计算 IIS
#     master.computeIIS()
#     master.write("master_model.ilp")


# print(vim[1][88])
# print(vim[4][22])
# print('over')
""" Column-and-constraint generation """
LB = -np.inf
UB = np.inf
max_iter = 100
max_iter_inner = 50
delta = 0.001
delta_a = 0.001
delta_b = 0.001
Gap = np.inf

''' create the subproblem '''
subProblem = Model('sub problem')
# close the outputflag
subProblem.setParam('Outputflag', 0)
'''subProblem Input from master'''
qimk_sol = {}
for i in range(retailer_number):
    qimk_sol[i] = {}
    for m in range(product_kind):
        qimk_sol[i][m] = {}
        for k in range(vehicle_number):
            qimk_sol[i][m][k] = qimk[i][m][k].x
            # if qimk[i][m][k].x > smallS:
            # print(f'initial_qimk[{i}][{m}][{k}].x:{qimk[i][m][k].x}')

'''create subProblem variables'''
epsilon_im = {}
Irim = {}
uim = {}
sim = {}
wijm = {}
zij = {}
ksi_im = {}
dim_prime = {}
# miu_im = {}

# epsilon_im Irim uim sim ksi miu
for i in range(retailer_number):
    epsilon_im[i] = {}
    Irim[i] = {}
    uim[i] = {}
    sim[i] = {}
    ksi_im[i] = {}
    dim_prime[i] = {}
    # miu_im[i] = {}
    for m in range(product_kind):
        epsilon_im[i][m] = subProblem.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f'epsilon_{i + 1}_{m + 1}')
        Irim[i][m] = subProblem.addVar(vtype=GRB.CONTINUOUS, name=f'Irim_{i + 1}_{m + 1}')
        uim[i][m] = subProblem.addVar(vtype=GRB.BINARY, name=f'uim_{i + 1}_{m + 1}')
        sim[i][m] = subProblem.addVar(lb=0, vtype=GRB.BINARY, name=f'sim_{i + 1}_{m + 1}')
        ksi_im[i][m] = subProblem.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'ksi_{i + 1}_{m + 1}')
        dim_prime[i][m] = subProblem.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'dim_prime{i + 1}_{m + 1}')
        # miu_im[i][m] = subProblem.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f'miu_{i + 1}_{m + 1}')

# wijm
for i in range(retailer_number):
    wijm[i] = {}
    for j in range(retailer_number):
        if i != j:
            wijm[i][j] = {}
            for m in range(product_kind):
                wijm[i][j][m] = subProblem.addVar(lb=0, vtype=GRB.CONTINUOUS,
                                                  name=f'w_{i + 1}_{j + 1}_{m + 1}')
# zij
for i in range(retailer_number):
    zij[i] = {}
    for j in range(retailer_number):
        if i != j:
            zij[i][j] = subProblem.addVar(vtype=GRB.BINARY, name=f'z_{i + 1}_{j + 1}')
phi = subProblem.addVar(vtype=GRB.CONTINUOUS, name='phi')

""" set objective """
sub_obj = LinExpr()
sub_obj.addTerms(1, phi)

# for i in range(retailer_number):
#     for m in range(product_kind):
#         for j in range(retailer_number):
#             if i !=j:
#                 sub_obj.addTerms(-pim[i][m],wijm[j][i][m])
subProblem.setObjective(sub_obj, GRB.MINIMIZE)
""" add constraints to subproblem """
# cons 1
pim_wijm_prosum = LinExpr()
for m in range(product_kind):
    for i in range(retailer_number):
        for j in range(retailer_number):
            if i != j:
                pim_wijm_prosum.addTerms(pim[i][m], wijm[i][j][m])
cij_zij_prosum = LinExpr()
for i in range(retailer_number):
    for j in range(retailer_number):
        if i != j:
            cij_zij_prosum.addTerms(-beta[iter_cnt] * cij_trans[i][j], zij[i][j])
him_wijm_produm = LinExpr()
for m in range(product_kind):
    for i in range(retailer_number):
        for j in range(retailer_number):
            if i != j:
                him_wijm_produm.addTerms(-beta[iter_cnt] * him[i][m], wijm[i][j][m])

subProblem.addConstr(phi >= pim_wijm_prosum + cij_zij_prosum + him_wijm_produm, name='subObj_cons')

# cons 17 real demand uncertainty and real inventory calculate
epsilon_im_sum = LinExpr()
for i in range(retailer_number):
    for m in range(product_kind):
        epsilon_im_sum.addTerms(1, epsilon_im[i][m])
        qimk_sol_k_sum = 0
        for k in range(vehicle_number):
            qimk_sol_k_sum += qimk_sol[i][m][k]
        subProblem.addConstr(
            Irim[i][m] == Iim0[i + 1][m] + qimk_sol_k_sum - dim_prime[i][m], \
            name=f'real_inventory_cons_i{i}_m{m + 1}')
        subProblem.addConstr(dim_prime[i][m] <= dim[i][m] + epsilon_im[i][m] * rho * dim[i][m],
                             name=f'uncertain_demand_cons_i{i}_m{m + 1}')
        # subProblem.addConstr(Irim[i][m] - (Iim0[i + 1][m] + qimk_sol_k_sum - dim[i][m] * epsilon_im[i][m]) <= smallS,
        #                      name=f'real_inventory_cons_i{i}_m{m + 1}_1')
        # subProblem.addConstr(Irim[i][m] - (Iim0[i + 1][m] + qimk_sol_k_sum - dim[i][m] * epsilon_im[i][m]) >= -smallS,
        #                      name=f'real_inventory_cons_i{i}_m{m + 1}_2')
# subProblem.addConstr(epsilon_im_sum <= tau_T)
# Add global budget constraint (E_total)
subProblem.addConstr(quicksum(epsilon_im[i][m] for i in epsilon_im.keys() for m in epsilon_im[i].keys()) <= Gamma,
                     name="Global_Budget")

# Add complementary product constraints (E_comp)
for node, (products, relation) in node_product_relationships.items():
    if relation == "comp":
        subProblem.addConstr(quicksum(epsilon_im[node][m] for m in products) <= Gamma_comp[node],
                             name=f"Complementary_{node}")

# Add synchronous product constraints (E_sync)
for node, (products, relation) in node_product_relationships.items():
    if relation == "sync":
        for l in range(len(products) - 1):
            m_l = products[l]
            m_l_plus_1 = products[l + 1]
            subProblem.addConstr(epsilon_im[node][m_l] - epsilon_im[node][m_l_plus_1] <= Gamma_sync[node],
                                 name=f"Sync_{node}_{m_l}_{m_l_plus_1}")
            subProblem.addConstr(epsilon_im[node][m_l_plus_1] - epsilon_im[node][m_l] <= Gamma_sync[node],
                                 name=f"Sync_{node}_{m_l_plus_1}_{m_l}")

# Add conflict constraints (E_conflict)
for product, (nodes, relation) in product_node_relationships.items():
    if relation == "conflict":
        subProblem.addConstr(quicksum(epsilon_im[i][product] for i in nodes) <= Gamma_conflict[product],
                             name=f"Conflict_{product}")

# Add regional constraints (E_region)
for product, (nodes, relation) in product_node_relationships.items():
    if relation == "region":
        for l in range(len(nodes) - 1):
            i_l = nodes[l]
            i_l_plus_1 = nodes[l + 1]
            subProblem.addConstr(epsilon_im[i_l][product] - epsilon_im[i_l_plus_1][product] <= Gamma_region[product],
                                 name=f"Region_{product}_{i_l}_{i_l_plus_1}")
            subProblem.addConstr(epsilon_im[i_l_plus_1][product] - epsilon_im[i_l][product] <= Gamma_region[product],
                                 name=f"Region_{product}_{i_l_plus_1}_{i_l}")

# 需求满足率要求约束 ---DSR0---
# for i in range(retailer_number):
#     for m in range(product_kind):
#         wjim_j_sum=LinExpr()
#         for j in range(retailer_number):
#             if i !=j:
#                 wjim_j_sum.addTerms(1,wijm[j][i][m])
#                 wjim_j_sum.addTerms(-1,wijm[i][j][m])
#         qimk_sol_k_sum = 0
#         for k in range(vehicle_number):
#             qimk_sol_k_sum += qimk_sol[i][m][k]
#         subProblem.addConstr(Iim0[i][m]+wjim_j_sum+qimk_sol_k_sum >= 0.9 * dim[i][m] * epsilon_im[i][m])

# cons 20-23:uim sim definition
for i in range(retailer_number):
    for m in range(product_kind):
        subProblem.addConstr(Irim[i][m] <= bigM * sim[i][m])
        subProblem.addConstr(Irim[i][m] + bigM * (1 - sim[i][m]) >= 0)
        subProblem.addConstr(Irim[i][m] <= bigM * (1 - uim[i][m]))
        subProblem.addConstr(Irim[i][m] + bigM * uim[i][m] >= 0)
# cons 25-30 ksi definition
for i in range(retailer_number):
    for m in range(product_kind):
        subProblem.addConstr(ksi_im[i][m] >= Irim[i][m] - bigM * (1 - sim[i][m]))
        subProblem.addConstr(ksi_im[i][m] <= Irim[i][m] + bigM * (1 - sim[i][m]))
        subProblem.addConstr(ksi_im[i][m] <= bigM * sim[i][m])
        wijm_j_sum = LinExpr()
        wjim_j_sum = LinExpr()
        # zij_j_sum_M = LinExpr()
        for j in range(retailer_number):
            if i != j:
                wijm_j_sum.addTerms(1, wijm[i][j][m])
                wjim_j_sum.addTerms(1, wijm[j][i][m])
                # zij_j_sum_M.addTerms(bigM, zij[i][j])
        subProblem.addConstr(wijm_j_sum <= ksi_im[i][m])
        subProblem.addConstr(wjim_j_sum <= Ui[i + 1] * uim[i][m])
        # subProblem.addConstr(wijm_j_sum <= zij_j_sum_M)
for i in range(retailer_number):
    for j in range(retailer_number):
        if i != j:
            wijm_m_sum = LinExpr()
            for m in range(product_kind):
                wijm_m_sum.addTerms(1, wijm[i][j][m])
            subProblem.addConstr(wijm_m_sum <= Ui[j + 1] * zij[i][j])

# # cons 32-37 miu definition
# for i in range(retailer_number):
#     for m in range(product_kind):
#         # subProblem.addConstr(miu_im[i][m] >= -Irim[i][m] - bigM * (1 - uim[i][m]))
#         # subProblem.addConstr(miu_im[i][m] <= -Irim[i][m] + bigM * (1 - uim[i][m]))
#         # subProblem.addConstr(miu_im[i][m] <= bigM * uim[i][m])
#         wjim_j_sum = LinExpr()
#         zji_j_sum_M = LinExpr()
#         for j in range(retailer_number):
#             if i != j:
#                 wjim_j_sum.addTerms(1, wijm[j][i][m])
#                 zji_j_sum_M.addTerms(bigM, zij[j][i])
#         # subProblem.addConstr(wjim_j_sum <= miu_im[i][m])---------------------------------------------------------------------------------------------------
#         subProblem.addConstr(wjim_j_sum <= zji_j_sum_M)

# cons38-39
for i in range(retailer_number):
    zij_j_sum = LinExpr()
    zji_j_sum = LinExpr()
    for j in range(retailer_number):
        if i != j:
            zij_j_sum.addTerms(1, zij[i][j])
            zji_j_sum.addTerms(1, zij[j][i])
    sim_m_sum = LinExpr()
    uim_m_sum = LinExpr()
    for m in range(product_kind):
        sim_m_sum.addTerms(1, sim[i][m])
        uim_m_sum.addTerms(1, uim[i][m])
    subProblem.addConstr(zij_j_sum <= sim_m_sum)
    subProblem.addConstr(zji_j_sum <= uim_m_sum)

# cons 40
for i in range(retailer_number):
    Irim_m_sum = LinExpr()
    wijm_j_sum = LinExpr()
    wjim_j_sum = LinExpr()
    for m in range(product_kind):
        Irim_m_sum.addTerms(vim[i][m], Irim[i][m])
        for j in range(retailer_number):
            if i != j:
                wijm_j_sum.addTerms(vim[i][m], wijm[i][j][m])
                wjim_j_sum.addTerms(vim[i][m], wijm[j][i][m])
    subProblem.addConstr(Irim_m_sum - wijm_j_sum + wjim_j_sum <= Ui[i + 1])

# # cons 41
# for i in range(retailer_number):
#     psd_m_sum = LinExpr()
#     peps_m_sum = LinExpr()
#     for m in range(product_kind):
#         peps_m_sum.addTerms(DSR0* pim[i][m] * dim[i][m], epsilon_im[i][m])
#         psd_m_sum.addTerms(pim[i][m], sd_im[i,m])
#     subProblem.addConstr(psd_m_sum - peps_m_sum >= 0)

""" subProblem.optimize """
subProblem.Params.TimeLimit = 600
subProblem.Params.NodefileStart = 0.5
subProblem.optimize()

# 如果模型不可行，计算 IIS
if subProblem.Status == GRB.INFEASIBLE or master.Status == GRB.INF_OR_UNBD:
    print("subProblem Model is infeasible.")
    # 计算 IIS
    # subProblem.computeIIS()
    # subProblem.write("initial subProblem.ilp")

iter_b0 = 0
while (phi.x > delta_0 and iter_b0 < iter_max_0) or iter_b0 == 0:
    # calculate beta
    subProblem_numerator = 0
    subProblem_denominator = 0
    for m in range(product_kind):
        for i in range(retailer_number):
            for j in range(retailer_number):
                if i != j:
                    subProblem_numerator += pim[i][m] * wijm[i][j][m].x
    for i in range(retailer_number):
        for j in range(retailer_number):
            if i != j:
                subProblem_denominator += cij_trans[i][j] * zij[i][j].x
    for m in range(product_kind):
        for i in range(retailer_number):
            for j in range(retailer_number):
                if i != j:
                    subProblem_denominator += him[i][m] * wijm[i][j][m].x
    if subProblem_denominator != 0:
        beta[iter_cnt] = subProblem_numerator / subProblem_denominator
    else:
        beta[iter_cnt] = 0
    # print('内循环第{}代：beta={}。'.format(iter_pab, beta[iter_cnt]))
    iter_b0 += 1

    """ reSet subproblem objcons """
    subProblem.remove(subProblem.getConstrByName('subObj_cons'))
    pim_wijm_prosum = LinExpr()
    for m in range(product_kind):
        for i in range(retailer_number):
            for j in range(retailer_number):
                if i != j:
                    pim_wijm_prosum.addTerms(pim[i][m], wijm[i][j][m])
    cij_zij_prosum = LinExpr()
    for i in range(retailer_number):
        for j in range(retailer_number):
            if i != j:
                cij_zij_prosum.addTerms(-beta[iter_cnt] * cij_trans[i][j], zij[i][j])
    him_wijm_produm = LinExpr()
    for m in range(product_kind):
        for i in range(retailer_number):
            for j in range(retailer_number):
                if i != j:
                    him_wijm_produm.addTerms(-beta[iter_cnt] * him[i][m], wijm[i][j][m])
    subProblem.addConstr(phi >= pim_wijm_prosum + cij_zij_prosum + him_wijm_produm, name='subObj_cons')
    # solve the resulted sub problem
    subProblem.optimize()
    # print("phi.x:",phi.x)

# subProblem.write('SP.lp')

# print('\n\n\n *******            C&CG starts          *******  ')
# print('\n **                Initial Solution             ** \n')
# 如果模型不可行，计算 IIS
# if master.Status.Status == GRB.INFEASIBLE or master.Status == GRB.INF_OR_UNBD:
#     print("master Model is infeasible.")
#     # 计算 IIS
#     master.computeIIS()
#     master.write("master_model.ilp")
# else:
#     print('*' * 100 + '\n(master)Obj = {}'.format(master.objVal))
#     for k in range(vehicle_number):
#         for i in range(retailer_number):
#             for m in range(product_kind):
#                 if qimk[i][m][k].x != 0:
#                     print(f'第{k + 1}辆车向节点{i + 1}配送商品{m + 1}量：{qimk[i][m][k].x}')

# if (subProblem.status != 2):
#     print('The subProblem is infeasible or unbounded!')
#     print('Status: {}'.format(subProblem.status))
#     # 计算 IIS
#     subProblem.computeIIS()
#     subProblem.write("subProblem_model.ilp")
# else:
# print('Obj(sub) : {}'.format(subProblem.ObjVal), end='\t | \n')
# for m in range(product_kind):
#     for i in range(retailer_number):
#         for j in range(retailer_number):
#             if i != j:
#                 if wijm[i][j][m].x != 0:
#                     print(f'节点{i + 1}向节点{j + 1}转运商品{m + 1}量为：{wijm[i][j][m].x}')
# for i in range(retailer_number):
#     for m in range(product_kind):
#         # print(f'节点{i + 1}处商品{m + 1}的实际需求不确定参数为：{epsilon_im[i][m].x}')
#         epsilon_master[iter_cnt, i, m] = epsilon_im[i][m].x
# if ksi_im[i][m].x != 0:
#     print(f'节点{i+1}处商品{m + 1}有余量为：{ksi_im[i][m].x}')
# if miu_im[i][m].x != 0:
#     print(f'节点{i+1}处商品{m + 1}缺货：{miu_im[i][m].x}')
# solution = {var.VarName: var.x for var in subProblem.getVars()}
# print(solution)
# for key, value in solution.items():
#     if value != 0:
#         print(f'variable：{key}---{value}')

epsilon_master = {}
for i in range(retailer_number):
    for m in range(product_kind):
        # print(f'节点{i + 1}处商品{m + 1}的实际需求不确定参数为：{epsilon_im[i][m].x}')
        epsilon_master[iter_cnt, i, m] = epsilon_im[i][m].x
        # print(f'epsilon_im[{i}][{m}].x:{epsilon_im[i][m].x}')

# print(epsilon_master)
""" 
 Update the initial Lower bound 
"""
# LB = max(LB, alpha[iter_cnt] + beta[iter_cnt])
# print('LB (iter {}): {}'.format(iter_cnt, LB))

# close the outputflag
master.setParam('Outputflag', 0)
subProblem.setParam('Outputflag', 0)

'''create new variables to master from subProblem'''
wijm_master = {}
zij_master = {}
Irim_master = {}
uim_master = {}
sim_master = {}
ksi_im_master = {}
miu_im_master = {}
dim_prime_master = {}

beta_master = {}
# beta[iter_cnt] = beta[iter_b0]
# alpha[iter_cnt] = alpha[iter_a0]
# print(alpha[iter_cnt])
# print(beta[iter_cnt])
beta_master[iter_cnt] = 1

"""
 Main loop of CCG algorithm 
"""
gap_cpr = None
not_change_times = 0
max_not_change_time = 2
while (abs(UB - LB) > delta) and (iter_cnt <= max_iter):
    # print('\n iter : {} '.format(iter_cnt), ' |*********** Main loop of CCG algorithm *************   \n')

    for i in range(retailer_number):
        for j in range(retailer_number):
            if i != j:
                for m in range(product_kind):
                    wijm_master[iter_cnt, i, j, m] = master.addVar(lb=0
                                                                   , vtype=GRB.CONTINUOUS
                                                                   ,
                                                                   name=f'master[{iter_cnt}]_w_{i + 1}_{j + 1}_{m + 1}')
    for i in range(retailer_number):
        for j in range(retailer_number):
            if i != j:
                zij_master[iter_cnt, i, j] = master.addVar(vtype=GRB.BINARY,
                                                           name=f'master[{iter_cnt}]_z_{i + 1}_{j + 1}')
    for i in range(retailer_number):
        for m in range(product_kind):
            # epsilon_master[iter_cnt,i,m] = subProblem.addVar(lb=0, ub=2, vtype=GRB.CONTINUOUS, name=f'master[{iter_cnt}]_epsilon_{i + 1}_{m + 1}')
            Irim_master[iter_cnt, i, m] = master.addVar(vtype=GRB.CONTINUOUS,
                                                        name=f'master[{iter_cnt}]_Irim_{i + 1}_{m + 1}')
            uim_master[iter_cnt, i, m] = master.addVar(vtype=GRB.BINARY, name=f'master[{iter_cnt}]_uim_{i + 1}_{m + 1}')
            sim_master[iter_cnt, i, m] = master.addVar(lb=0, vtype=GRB.BINARY,
                                                       name=f'master[{iter_cnt}]_sim_{i + 1}_{m + 1}')
            ksi_im_master[iter_cnt, i, m] = master.addVar(lb=0, vtype=GRB.CONTINUOUS,
                                                          name=f'master[{iter_cnt}]_ksi_{i + 1}_{m + 1}')
            miu_im_master[iter_cnt, i, m] = master.addVar(lb=0, vtype=GRB.CONTINUOUS,
                                                          name=f'master[{iter_cnt}]_miu_{i + 1}_{m + 1}')
            dim_prime_master[iter_cnt, i, m] = master.addVar(lb=0, vtype=GRB.CONTINUOUS,
                                                             name=f'master[{iter_cnt}]_dim_prime_{i + 1}_{m + 1}')
    # create worst case related constraints
    # cons 17 real demand uncertainty and real inventory calculate------------------master_add_sub_cons
    for i in range(retailer_number):
        for m in range(product_kind):
            qimk_master_k_sum = LinExpr()
            for k in range(vehicle_number):
                qimk_master_k_sum.addTerms(1, qimk[i][m][k])
            master.addConstr(
                Irim_master[iter_cnt, i, m] ==
                Iim0[i + 1][m] + qimk_master_k_sum - dim_prime_master[iter_cnt, i, m])
            master.addConstr(
                dim_prime_master[iter_cnt, i, m] <= (dim[i][m] + epsilon_master[iter_cnt, i, m] * rho * dim[i][m]),
                name=f'uncertain_demand_cons_i{i}_m{m + 1}')

            # master.addConstr(
            #     Irim_master[iter_cnt, i, m] - (
            #                 Iim0[i + 1][m] + qimk_master_k_sum - dim[i][m] * epsilon_master[iter_cnt, i, m]) <= smallS)
            # master.addConstr(
            #     Irim_master[iter_cnt, i, m] - (
            #                 Iim0[i + 1][m] + qimk_master_k_sum - dim[i][m] * epsilon_master[iter_cnt, i, m]) >= -smallS)
    # print("epsilon_master",epsilon_master)
    # 需求满足率要求约束---DSR0---
    # for i in range(retailer_number):
    #     for m in range(product_kind):
    #         wjim_j_sum = LinExpr()
    #         for j in range(retailer_number):
    #             if i != j:
    #                 wjim_j_sum.addTerms(1, wijm_master[iter_cnt,j,i,m])
    #                 wjim_j_sum.addTerms(-1, wijm_master[iter_cnt,i,j,m])
    #         qimk_k_sum = LinExpr()
    #         for k in range(vehicle_number):
    #             qimk_k_sum.addTerms(1,qimk[i][m][k])
    #         master.addConstr(Iim0[i][m] + wjim_j_sum + qimk_k_sum >= 0.9 * dim[i][m] *epsilon_master[iter_cnt,i,m])

    # cons 20-23:uim sim definition------------------master_add_sub_cons
    for i in range(retailer_number):
        for m in range(product_kind):
            master.addConstr(Irim_master[iter_cnt, i, m] <= bigM * sim_master[iter_cnt, i, m])
            master.addConstr(Irim_master[iter_cnt, i, m] + bigM * (1 - sim_master[iter_cnt, i, m]) >= 0)
            master.addConstr(Irim_master[iter_cnt, i, m] <= bigM * (1 - uim_master[iter_cnt, i, m]))
            master.addConstr(Irim_master[iter_cnt, i, m] + bigM * uim_master[iter_cnt, i, m] >= 0)
    # cons 25-30 ksi definition------------------master_add_sub_cons
    for i in range(retailer_number):
        for m in range(product_kind):
            master.addConstr(ksi_im_master[iter_cnt, i, m] >= Irim_master[iter_cnt, i, m] - bigM * (
                    1 - sim_master[iter_cnt, i, m]))
            master.addConstr(ksi_im_master[iter_cnt, i, m] <= Irim_master[iter_cnt, i, m] + bigM * (
                    1 - sim_master[iter_cnt, i, m]))
            master.addConstr(ksi_im_master[iter_cnt, i, m] <= bigM * sim_master[iter_cnt, i, m])
            wijm_j_sum = LinExpr()
            wjim_j_sum = LinExpr()
            # zij_j_sum_M = LinExpr()
            for j in range(retailer_number):
                if i != j:
                    wijm_j_sum.addTerms(1, wijm_master[iter_cnt, i, j, m])
                    wjim_j_sum.addTerms(1, wijm_master[iter_cnt, j, i, m])
                    # zij_j_sum_M.addTerms(bigM, zij_master[iter_cnt, i, j])
            master.addConstr(wijm_j_sum <= ksi_im_master[iter_cnt, i, m])
            master.addConstr(wjim_j_sum <= Ui[i + 1] * uim_master[iter_cnt, i, m])
            # master.addConstr(wijm_j_sum <= zij_j_sum_M)
    for i in range(retailer_number):
        for j in range(retailer_number):
            if i != j:
                wijm_m_sum = LinExpr()
                for m in range(product_kind):
                    wijm_m_sum.addTerms(1, wijm_master[iter_cnt, i, j, m])
                master.addConstr(wijm_m_sum <= Ui[j + 1] * zij_master[iter_cnt, i, j])

    # # cons 32-37 miu definition------------------master_add_sub_cons
    # for i in range(retailer_number):
    #     for m in range(product_kind):
    #         # master.addConstr(miu_im_master[iter_cnt, i, m] >= -Irim_master[iter_cnt, i, m] - bigM * (
    #         #         1 - uim_master[iter_cnt, i, m]))
    #         # master.addConstr(miu_im_master[iter_cnt, i, m] <= -Irim_master[iter_cnt, i, m] + bigM * (
    #         #         1 - uim_master[iter_cnt, i, m]))
    #         # master.addConstr(miu_im_master[iter_cnt, i, m] <= bigM * uim_master[iter_cnt, i, m])
    #         wjim_j_sum = LinExpr()
    #         zji_j_sum_M = LinExpr()
    #         for j in range(retailer_number):
    #             if i != j:
    #                 wjim_j_sum.addTerms(1, wijm_master[iter_cnt, j, i, m])
    #                 zji_j_sum_M.addTerms(bigM, zij_master[iter_cnt, j, i])
    #         # master.addConstr(wjim_j_sum <= miu_im_master[iter_cnt, i, m])
    #         master.addConstr(wjim_j_sum <= zji_j_sum_M)

    # cons38-39------------------master_add_sub_cons
    for i in range(retailer_number):
        zij_j_sum = LinExpr()
        zji_j_sum = LinExpr()
        for j in range(retailer_number):
            if i != j:
                zij_j_sum.addTerms(1, zij_master[iter_cnt, i, j])
                zji_j_sum.addTerms(1, zij_master[iter_cnt, j, i])
        sim_m_sum_M = LinExpr()
        uim_m_sum_M = LinExpr()
        for m in range(product_kind):
            sim_m_sum_M.addTerms(1, sim_master[iter_cnt, i, m])
            uim_m_sum_M.addTerms(1, uim_master[iter_cnt, i, m])
        master.addConstr(zij_j_sum <= sim_m_sum_M)
        master.addConstr(zji_j_sum <= uim_m_sum_M)

    # cons 40------------------master_add_sub_cons
    for i in range(retailer_number):
        Irim_m_sum = LinExpr()
        wijm_j_sum = LinExpr()
        wjim_j_sum = LinExpr()
        for m in range(product_kind):
            Irim_m_sum.addTerms(vim[i][m], Irim_master[iter_cnt, i, m])
            for j in range(retailer_number):
                if i != j:
                    wijm_j_sum.addTerms(vim[i][m], wijm_master[iter_cnt, i, j, m])
                    wjim_j_sum.addTerms(vim[i][m], wijm_master[iter_cnt, j, i, m])
        master.addConstr(Irim_m_sum - wijm_j_sum + wjim_j_sum <= Ui[i + 1])

    # # cons 41------------------master_add_sub_cons
    # for i in range(retailer_number):
    #     psd_m_sum = LinExpr()
    #     peps_m_sum = LinExpr()
    #     for m in range(product_kind):
    #         peps_m_sum.addTerms(DSR0 * pim[i][m] * dim[i][m], epsilon_master[iter_cnt,i,m])
    #         psd_m_sum.addTerms(pim[i][m], sd_im_master[iter_cnt,i, m])
    #     master.addConstr(psd_m_sum - peps_m_sum >= 0)
    # if subproblem is frasible and bound, create variables xk+1 and add the new constraints
    if (subProblem.status == 2):
        """ add new sub constraints to master problem """
        # cons 1------------------master_add_sub_cons
        pim_wijm_prosum = LinExpr()
        for m in range(product_kind):
            for i in range(retailer_number):
                for j in range(retailer_number):
                    if i != j:
                        pim_wijm_prosum.addTerms(pim[i][m], wijm_master[iter_cnt, i, j, m])
        cij_zij_prosum = LinExpr()
        for i in range(retailer_number):
            for j in range(retailer_number):
                if i != j:
                    cij_zij_prosum.addTerms(-beta_master[iter_cnt] * cij_trans[i][j], zij_master[iter_cnt, i, j])
        him_wijm_produm = LinExpr()
        for m in range(product_kind):
            for i in range(retailer_number):
                for j in range(retailer_number):
                    if i != j:
                        him_wijm_produm.addTerms(-beta_master[iter_cnt] * him[i][m], wijm_master[iter_cnt, i, j, m])
        master.addConstr(eta <= pim_wijm_prosum + cij_zij_prosum + him_wijm_produm,
                         name=f'sub_eta_maxmin_cons_{iter_cnt}')
        # master.addConstr(delta >= pim_wijm_prosum + cij_zij_prosum + him_wijm_produm, name=f'sub_eta_maxmin_cons')

        # now_max=0
        # for new_iter in range(iter_cnt):
        #     master_sub_numerator = 0
        #     master_sub_denominator = 0
        #     for m in range(product_kind):
        #         for i in range(retailer_number):
        #             for j in range(retailer_number):
        #                 if i != j:
        #                     master_sub_numerator += pim[i][m] * wijm_master[new_iter, i, j, m].x
        #     for i in range(retailer_number):
        #         for j in range(retailer_number):
        #             if i != j:
        #                 master_sub_denominator += cij_trans[i][j] * zij_master[new_iter, i, j].x
        #     for m in range(product_kind):
        #         for i in range(retailer_number):
        #             for j in range(retailer_number):
        #                 if i != j:
        #                     master_sub_denominator += him[i][m] * wijm_master[new_iter, i, j, m].x
        #     if master_sub_numerator-beta_master[new_iter]*master_sub_denominator>=now_max:
        #         now_max=master_sub_numerator-beta_master[new_iter]*master_sub_denominator
        #     if master_sub_denominator != 0:
        #          beta_master[iter_cnt+1]== master_sub_numerator / master_sub_denominator
        #     else:
        #         beta_master[new_iter] = 0
        # print('内循环第{}代：alpha={},beta_master={}。'.format(iter_paa, alpha[new_iter], beta_master[new_iter]))

        # for new_i in range(iter_cnt):
        #     '''reset constraint with beta'''
        #     master.remove(master.getConstrByName(f'sub_eta_maxmin_cons_{new_i}'))
        #     pim_wijm_prosum = LinExpr()
        #     for m in range(product_kind):
        #         for i in range(retailer_number):
        #             for j in range(retailer_number):
        #                 if i != j:
        #                     pim_wijm_prosum.addTerms(pim[i][m], wijm_master[new_i, i, j, m])
        #     cij_zij_prosum = LinExpr()
        #     for i in range(retailer_number):
        #         for j in range(retailer_number):
        #             if i != j:
        #                 cij_zij_prosum.addTerms(-beta_master[new_i] * cij_trans[i][j], zij_master[new_i, i, j])
        #     him_wijm_produm = LinExpr()
        #     for m in range(product_kind):
        #         for i in range(retailer_number):
        #             for j in range(retailer_number):
        #                 if i != j:
        #                     him_wijm_produm.addTerms(-beta_master[new_i] * him[i][m], wijm_master[iter_cnt, i, j, m])
        #     master.addConstr(eta <= pim_wijm_prosum + cij_zij_prosum + him_wijm_produm, name=f'sub_eta_maxmin_cons_{new_i}')

    master.optimize()
    # master.write('master.lp')
    # master.feasRelaxS(0,False,False,True)
    # master.write('feasmaster.lp')
    # master.write('feasi_master_result.sol')
    if (master.status != 2):
        # print('The subProblem is infeasible or unbounded!')
        # print('Status: {}'.format(master.status))
        # # 计算 IIS
        # print(master.status)
        # master.computeIIS()
        # master.write("master_model.ilp")
        master.remove(master.getConstrByName(f'sub_eta_maxmin_cons_{iter_cnt}'))
        master.optimize()

    iter_paa = 0
    # while ((((master.ObjVal - eta.x) > delta_a) or (eta.x > delta_a)) and (iter_paa < max_iter_inner)) or (
    #         iter_paa == 0):
    while (((master.ObjVal - eta.x) > delta_a) and (iter_paa < max_iter_inner)) or (
            iter_paa == 0):
        if (master.ObjVal - eta.x) > delta_a:
            master_numerator = 0
            master_denominator = 0
            for i in range(retailer_number):
                for m in range(product_kind):
                    for k in range(vehicle_number):
                        master_numerator += pim[i][m] * qimk[i][m][k].x
            for i in range(nodes_number):
                for j in range(nodes_number):
                    if i < j:
                        for k in range(vehicle_number):
                            master_denominator += cij[i][j] * xijk[i][j][k].x
            for i in range(nodes_number):
                for m in range(product_kind):
                    master_denominator += him[i][m] * Iim[i][m].x
            for k in range(vehicle_number):
                master_denominator += bk[k] * yik[0][k].x
            if master_denominator == 0:
                alpha[iter_cnt] = 0
            else:
                alpha[iter_cnt] = master_numerator / master_denominator
            # print(f'alpha[iter_cnt]:{alpha[iter_cnt]},master_numerator:{master_numerator},master_denominator:{master_denominator}')

        # if eta.x>delta_a:
        #     master_sub_numerator = 0
        #     master_sub_denominator = 0
        #     for m in range(product_kind):
        #         for i in range(retailer_number):
        #             for j in range(retailer_number):
        #                 if i != j:
        #                     master_sub_numerator += pim[i][m] * wijm_master[iter_cnt, i, j, m].x
        #     for i in range(retailer_number):
        #         for j in range(retailer_number):
        #             if i != j:
        #                 master_sub_denominator += cij_trans[i][j] * zij_master[iter_cnt, i, j].x
        #     for m in range(product_kind):
        #         for i in range(retailer_number):
        #             for j in range(retailer_number):
        #                 if i != j:
        #                     master_sub_denominator += him[i][m] * wijm_master[iter_cnt, i, j, m].x
        #     if master_sub_denominator != 0:
        #         beta_master[iter_cnt] = master_sub_numerator / master_sub_denominator
        #     else:
        #         beta_master[iter_cnt] = 0
        #     print(f'beta_master[iter_cnt]:{beta_master[iter_cnt]},master_sub_numerator:{master_sub_numerator},master_sub_denominator:{master_sub_denominator}')
        # print('内循环第{}代：alpha={},beta_master={}。'.format(iter_paa, alpha[iter_cnt], beta_master[iter_cnt]))
        """ reSet master objective """
        master_obj = LinExpr()
        for i in range(retailer_number):
            for m in range(product_kind):
                for k in range(vehicle_number):
                    master_obj.addTerms(pim[i][m], qimk[i][m][k])
        for i in range(nodes_number):
            for j in range(nodes_number):
                if i < j:
                    for k in range(vehicle_number):
                        master_obj.addTerms(-alpha[iter_cnt] * cij[i][j], xijk[i][j][k])
        for i in range(nodes_number):
            for m in range(product_kind):
                master_obj.addTerms(-alpha[iter_cnt] * him[i][m], Iim[i][m])
        for k in range(vehicle_number):
            master_obj.addTerms(-alpha[iter_cnt] * bk[k], yik[0][k])
        master_obj.addTerms(1, eta)
        master.setObjective(master_obj, GRB.MAXIMIZE)

        for new_iter in range(iter_cnt):
            master_sub_numerator = 0
            master_sub_denominator = 0
            for m in range(product_kind):
                for i in range(retailer_number):
                    for j in range(retailer_number):
                        if i != j:
                            master_sub_numerator += pim[i][m] * wijm_master[new_iter, i, j, m].x
            for i in range(retailer_number):
                for j in range(retailer_number):
                    if i != j:
                        master_sub_denominator += cij_trans[i][j] * zij_master[new_iter, i, j].x
            for m in range(product_kind):
                for i in range(retailer_number):
                    for j in range(retailer_number):
                        if i != j:
                            master_sub_denominator += him[i][m] * wijm_master[new_iter, i, j, m].x
            if master_sub_numerator - beta_master[new_iter] * master_sub_denominator >= delta_a:
                if master_sub_denominator != 0:
                    beta_master[new_iter] = master_sub_numerator / master_sub_denominator
                else:
                    beta_master[new_iter] = 0
                '''reset constraint with beta'''
                try:
                    master.remove(master.getConstrByName(f'sub_eta_maxmin_cons_{new_iter}'))
                except gurobipy.GurobiError:
                    pass
                pim_wijm_prosum = LinExpr()
                for m in range(product_kind):
                    for i in range(retailer_number):
                        for j in range(retailer_number):
                            if i != j:
                                pim_wijm_prosum.addTerms(pim[i][m], wijm_master[new_iter, i, j, m])
                cij_zij_prosum = LinExpr()
                for i in range(retailer_number):
                    for j in range(retailer_number):
                        if i != j:
                            cij_zij_prosum.addTerms(-beta_master[new_iter] * cij_trans[i][j],
                                                    zij_master[new_iter, i, j])
                him_wijm_produm = LinExpr()
                for m in range(product_kind):
                    for i in range(retailer_number):
                        for j in range(retailer_number):
                            if i != j:
                                him_wijm_produm.addTerms(-beta_master[new_iter] * him[i][m],
                                                         wijm_master[iter_cnt, i, j, m])
                master.addConstr(eta <= pim_wijm_prosum + cij_zij_prosum + him_wijm_produm,
                                 name=f'sub_eta_maxmin_cons_{new_iter}')

        # '''reset constraint with beta'''
        # try:
        #     master.remove(master.getConstrByName(f'sub_eta_maxmin_cons_{iter_cnt}'))
        # except gurobipy.GurobiError:
        #     pass
        # pim_wijm_prosum = LinExpr()
        # for m in range(product_kind):
        #     for i in range(retailer_number):
        #         for j in range(retailer_number):
        #             if i != j:
        #                 pim_wijm_prosum.addTerms(pim[i][m], wijm_master[iter_cnt, i, j, m])
        # cij_zij_prosum = LinExpr()
        # for i in range(retailer_number):
        #     for j in range(retailer_number):
        #         if i != j:
        #             cij_zij_prosum.addTerms(-beta_master[iter_cnt] * cij_trans[i][j], zij_master[iter_cnt, i, j])
        # him_wijm_produm = LinExpr()
        # for m in range(product_kind):
        #     for i in range(retailer_number):
        #         for j in range(retailer_number):
        #             if i != j:
        #                 him_wijm_produm.addTerms(-beta_master[iter_cnt] * him[i][m], wijm_master[iter_cnt, i, j, m])
        # master.addConstr(eta <= pim_wijm_prosum + cij_zij_prosum + him_wijm_produm,
        #                  name=f'sub_eta_maxmin_cons_{iter_cnt}')

        # solve the resulted master problem
        master.optimize()
        # print(f'iter_paa:{iter_paa}, eta:{eta.x},master.ObjVal-eta.x={master.ObjVal - eta.x}')
        iter_paa += 1

        # if master.status!=2:
        #     print('master(3 unbound):{}'.format(master.Status))
        #     master.write()
        #     master.computeIIS()
        #     master.write("master_0421.ilp")
    # print('Obj(master): {}'.format(master.ObjVal), end='\t |')
    """ Update the LB """
    UB = min(alpha[iter_cnt] + beta_master[iter_cnt], UB)
    # print('  UB (iter {}): {}'.format(iter_cnt, UB), end='\t |\n ')
    # for k in range(vehicle_number):
    #     for i in range(retailer_number):
    #         for m in range(product_kind):
    #             if qimk[i][m][k].x != 0:
    #                 print(f'第{k + 1}辆车向节点{i + 1}配送商品{m + 1}量：{qimk[i][m][k].x}')
    # print("eat:",eta.x)
    # print("alpha[iter_cnt]:",alpha[iter_cnt])
    # print("beta_master[iter_cnt]:",beta_master[iter_cnt])
    """ Update the subproblem """
    # first, get qimk_sol from updated master problem
    qimk_sol = {}
    for i in range(retailer_number):
        qimk_sol[i] = {}
        for m in range(product_kind):
            qimk_sol[i][m] = {}
            for k in range(vehicle_number):
                qimk_sol[i][m][k] = qimk[i][m][k].x
                # print(f'qimk[{i}][{m}][{k}].x:{qimk[i][m][k].x}')

    # change the coefficient of subproblem
    for i in range(retailer_number):
        for m in range(product_kind):
            subProblem.remove(subProblem.getConstrByName(f'real_inventory_cons_i{i}_m{m + 1}'))
            # subProblem.remove(subProblem.getConstrByName(f'real_inventory_cons_i{i}_m{m + 1}_1'))
            # subProblem.remove(subProblem.getConstrByName(f'real_inventory_cons_i{i}_m{m + 1}_2'))

    # cons 17 real demand uncertainty and real inventory calculate
    epsilon_im_sum = LinExpr()
    for i in range(retailer_number):
        for m in range(product_kind):
            epsilon_im_sum.addTerms(1, epsilon_im[i][m])
            qimk_sol_k_sum = 0
            for k in range(vehicle_number):
                qimk_sol_k_sum += qimk_sol[i][m][k]
            subProblem.addConstr(
                Irim[i][m] == Iim0[i + 1][m] + qimk_sol_k_sum - dim_prime[i][m], \
                name=f'real_inventory_cons_i{i}_m{m + 1}')
            subProblem.addConstr(dim_prime[i][m] <= dim[i][m] + epsilon_im[i][m] * rho * dim[i][m],
                                 name=f'uncertain_demand_cons_i{i}_m{m + 1}')

    # Add global budget constraint (E_total)
    subProblem.addConstr(
        quicksum(epsilon_im[i][m] for i in epsilon_im.keys() for m in epsilon_im[i].keys()) <= Gamma,
        name="Global_Budget")

    # Add complementary product constraints (E_comp)
    for node, (products, relation) in node_product_relationships.items():
        if relation == "comp":
            subProblem.addConstr(quicksum(epsilon_im[node][m] for m in products) <= Gamma_comp[node],
                                 name=f"Complementary_{node}")

    # Add synchronous product constraints (E_sync)
    for node, (products, relation) in node_product_relationships.items():
        if relation == "sync":
            for l in range(len(products) - 1):
                m_l = products[l]
                m_l_plus_1 = products[l + 1]
                subProblem.addConstr(epsilon_im[node][m_l] - epsilon_im[node][m_l_plus_1] <= Gamma_sync[node],
                                     name=f"Sync_{node}_{m_l}_{m_l_plus_1}")
                subProblem.addConstr(epsilon_im[node][m_l_plus_1] - epsilon_im[node][m_l] <= Gamma_sync[node],
                                     name=f"Sync_{node}_{m_l_plus_1}_{m_l}")

    # Add conflict constraints (E_conflict)
    for product, (nodes, relation) in product_node_relationships.items():
        if relation == "conflict":
            subProblem.addConstr(quicksum(epsilon_im[i][product] for i in nodes) <= Gamma_conflict[product],
                                 name=f"Conflict_{product}")

    # Add regional constraints (E_region)
    for product, (nodes, relation) in product_node_relationships.items():
        if relation == "region":
            for l in range(len(nodes) - 1):
                i_l = nodes[l]
                i_l_plus_1 = nodes[l + 1]
                subProblem.addConstr(
                    epsilon_im[i_l][product] - epsilon_im[i_l_plus_1][product] <= Gamma_region[product],
                    name=f"Region_{product}_{i_l}_{i_l_plus_1}")
                subProblem.addConstr(
                    epsilon_im[i_l_plus_1][product] - epsilon_im[i_l][product] <= Gamma_region[product],
                    name=f"Region_{product}_{i_l_plus_1}_{i_l}")

            # subProblem.addConstr(
            #     Irim[i][m] - (Iim0[i + 1][m] + qimk_sol_k_sum -dim[i][m] * epsilon_im[i][m]) <= smallS,
            #     name=f'real_inventory_cons_i{i}_m{m + 1}_1')
            # subProblem.addConstr(
            #     Irim[i][m] - (Iim0[i + 1][m] + qimk_sol_k_sum -dim[i][m] * epsilon_im[i][m]) >= -smallS,
            #     name=f'real_inventory_cons_i{i}_m{m + 1}_2')
    # print('all_beta:',beta)
    subProblem.optimize()
    iter_pab = 0
    # print('phi.x,iter_pab,iter_cnt:',phi.x,iter_pab,iter_cnt)
    # print('beta:',beta)
    while (phi.x > delta_b and iter_pab < max_iter_inner) or (iter_cnt == 0 and iter_pab == 0):
        # calculate beta
        subProblem_numerator = 0
        subProblem_denominator = 0
        for m in range(product_kind):
            for i in range(retailer_number):
                for j in range(retailer_number):
                    if i != j:
                        subProblem_numerator += pim[i][m] * wijm[i][j][m].x
        for i in range(retailer_number):
            for j in range(retailer_number):
                if i != j:
                    subProblem_denominator += cij_trans[i][j] * zij[i][j].x
        for m in range(product_kind):
            for i in range(retailer_number):
                for j in range(retailer_number):
                    if i != j:
                        subProblem_denominator += him[i][m] * wijm[i][j][m].x
        if subProblem_denominator != 0:
            beta[iter_cnt] = subProblem_numerator / subProblem_denominator
        else:
            beta[iter_cnt] = 0
        # print('内循环第{}代：beta={}。'.format(iter_pab, beta[iter_cnt]))
        # print('phi:',phi.x)

        """ reSet subproblem objcons """
        subProblem.remove(subProblem.getConstrByName('subObj_cons'))
        pim_wijm_prosum = LinExpr()
        for m in range(product_kind):
            for i in range(retailer_number):
                for j in range(retailer_number):
                    if i != j:
                        pim_wijm_prosum.addTerms(pim[i][m], wijm[i][j][m])
        cij_zij_prosum = LinExpr()
        for i in range(retailer_number):
            for j in range(retailer_number):
                if i != j:
                    cij_zij_prosum.addTerms(-beta[iter_cnt] * cij_trans[i][j], zij[i][j])
        him_wijm_produm = LinExpr()
        for m in range(product_kind):
            for i in range(retailer_number):
                for j in range(retailer_number):
                    if i != j:
                        him_wijm_produm.addTerms(-beta[iter_cnt] * him[i][m], wijm[i][j][m])
        subProblem.addConstr(phi >= pim_wijm_prosum + cij_zij_prosum + him_wijm_produm, name='subObj_cons')
        # solve the resulted sub problem
        subProblem.optimize()
        iter_pab += 1

    """ Update the lower bound """
    if (subProblem.status != 2):
        pass
        # print('The subProblem is infeasible or unbounded!')
        # print('Status: {}'.format(subProblem.status))
        # 计算 IIS
        # subProblem.computeIIS()
        # subProblem.write("subProblem_inner.ilp")
        # raise ValueError("subProblem.status != 2")
    else:
        # print('Obj(subProblem) : {}'.format(subProblem.ObjVal), end='\t\t\t | ')
        # for i in range(retailer_number):
        #     for j in range(retailer_number):
        #         if i != j:
        #             if zij[i][j].x != 0:
        #                 print(f'节点{i+1}向节点{j+1}转运')
        # for m in range(product_kind):
        #     for i in range(retailer_number):
        #         for j in range(retailer_number):
        #             if i != j:
        #                 if wijm[i][j][m].x != 0:
        #                     print(f'节点{i+1}向节点{j+1}转运商品{m + 1}量为：{wijm[i][j][m].x}')
        for i in range(retailer_number):
            for m in range(product_kind):
                # print(f'节点{i+1}处商品{m + 1}的实际需求不确定参数为：{epsilon_im[i][m].x}')
                epsilon_master[iter_cnt + 1, i, m] = epsilon_im[i][m].x
                # print(f'epsilon_im[{i}][{m}].x:{epsilon_im[i][m].x}')

                # if ksi_im[i][m].x != 0:
                #     print(f'节点{i+1}处商品{m + 1}有余量为：{ksi_im[i][m].x}')
                # if miu_im[i][m].x != 0:
                #     print(f'节点{i+1}处商品{m + 1}缺货：{miu_im[i][m].x}')
    """ 
     Update Lower bound 
    """
    LB = max(LB, alpha[iter_cnt] + beta[iter_cnt])
    # print('LB (iter {}): {}'.format(iter_cnt, LB), end='\t | \n')
    # for i in range(retailer_number):
    #     for m in range(product_kind):
    #         print(f'节点{i + 1}处商品{m + 1}的实际需求不确定参数为：{epsilon_im[i][m].x}')
    Gap = round(100 * (UB - LB) / UB, 2)
    alpha[iter_cnt + 1] = alpha[iter_cnt]
    beta_master[iter_cnt + 1] = beta_master[iter_cnt]
    beta[iter_cnt + 1] = beta[iter_cnt]
    iter_cnt += 1

    # print('eta = {}'.format(eta.x), end='\t | ')
    # print(' Gap: {} %  '.format(Gap), end='\t')
    # print('LB (iter {}): {}'.format(iter_cnt, LB), end='\t')
    # print('UB (iter {}): {}'.format(iter_cnt, UB), end='\t | \n')
    # print('*'*100)
    if Gap == gap_cpr:
        not_change_times += 1
    else:
        not_change_times = 0
    gap_cpr = Gap
    if not_change_times > max_not_change_time:
        break

# master.write('finalMP.lp')
# print('\n\nOptimal solution found !')
# print('Opt_Obj_alpha : {}'.format(alpha[iter_cnt]))
# print('Opt_Obj_beta : {}'.format(beta[iter_cnt]))
# print('Opt_Obj_beta_master : {}'.format(beta_master[iter_cnt]))
# print(' **  Final Gap: {} %  **  '.format(Gap))
if alpha[iter_cnt] != 0:
    final_Z1 = 1 / alpha[iter_cnt]
else:
    final_Z1 = 0
# print('final_Z1:{}'.format(final_Z1))
if beta[iter_cnt] != 0:
    final_Z2 = 1 / beta[iter_cnt]
else:
    final_Z2 = 0
# print('final_Z2:{}'.format(final_Z2))
# print('***  Final Solution(已还原的 min LR): {} ***'.format(final_Z1 + final_Z2))

# print(round(final_Z1 + final_Z2,3))

# print(round(final_Z1,3))


stage1_cost = 0
for i in range(nodes_number):
    for j in range(nodes_number):
        if i < j:
            for k in range(vehicle_number):
                stage1_cost += cij[i][j] * xijk[i][j][k].x
for i in range(nodes_number):
    for m in range(product_kind):
        stage1_cost += him[i][m] * Iim[i][m].x
for k in range(vehicle_number):
    stage1_cost += bk[k] * yik[0][k].x
# print('total cost (without qimk)')
# print(round(stage1_cost,2))
# print(round(Gap,4))
# print('\n  ********************  stage1  Solution  ********************  ')
# for k in range(vehicle_number):
#     for i in range(nodes_number):
#         for j in range(nodes_number):
#             if i < j:
#                 if xijk[i][j][k].x != 0:
#                     print(f'第{k + 1}辆车：从{i}到{j}')
# for i in range(retailer_number):
#     for m in range(product_kind):
#         for k in range(vehicle_number):
#             if qimk[i][m][k].x != 0:
#                 print(f'第{k + 1}辆车：向节点{i + 1}配送商品{m + 1}量：{qimk[i][m][k].x}')
# for m in range(product_kind):
#     print(f'配送中心进购产品{m + 1}量：{rm[m].x}')
#
# print('\n  ********************    stage2 solution  ********************  ')
# for i in range(retailer_number):
#     for j in range(retailer_number):
#         if i != j:
#             if zij[i][j].x != 0:
#                 print(f'节点{i + 1}向节点{j + 1}转运')
# for m in range(product_kind):
#     for i in range(retailer_number):
#         for j in range(retailer_number):
#             if i != j:
#                 if wijm[i][j][m].x != 0:
#                     print(f'节点{i + 1}向节点{j + 1}转运商品{m + 1}量为：{wijm[i][j][m].x}')
# for i in range(retailer_number):
#     for m in range(product_kind):
#         print(f'节点{i + 1}处商品{m + 1}的实际需求不确定参数为：{epsilon_im[i][m].x}')
#         if ksi_im[i][m].x != 0:
#             print(f'节点{i + 1}处商品{m + 1}有余量为：{ksi_im[i][m].x}')
#         if miu_im[i][m].x != 0:
#             print(f'节点{i + 1}处商品{m + 1}缺货：{miu_im[i][m].x}')

"""the model of checking stage2 with real demand"""
''' create the check model '''
check_model = Model('check_model')
# close the outputflag
check_model.setParam('Outputflag', 0)
'''check model Input from CCG-FP'''
qimk_check = {}
for i in range(retailer_number):
    for m in range(product_kind):
        for k in range(vehicle_number):
            qimk_check[i, m, k] = qimk[i][m][k].x

'''create parametres'''
Irim_check = {}
ksi_im_check = {}
miu_im_check = {}
for i in range(retailer_number):
    for m in range(product_kind):
        qimk_k_sum = 0
        for k in range(vehicle_number):
            qimk_k_sum += qimk_check[i, m, k]
        Irim_check[i, m] = Iim0[i + 1][m] + qimk_k_sum - dim_real[i][m]
        if Irim_check[i, m] >= 0:
            miu_im_check[i, m] = 0
            ksi_im_check[i, m] = Irim_check[i, m]
        else:
            miu_im_check[i, m] = -Irim_check[i, m]
            ksi_im_check[i, m] = 0

miu_positive = {}
ksi_positive = {}
for key, value in miu_im_check.items():
    if value != 0:
        miu_positive[key] = value
for key, value in ksi_positive.items():
    if value != 0:
        ksi_positive[key] = value

# print('ksi_positive')
# print(ksi_positive)
# print('miu_positive')
# print(miu_positive)


'''create variables'''
wijm_check = {}
zij_check = {}

# wijm
for i in range(retailer_number):
    wijm_check[i] = {}
    for j in range(retailer_number):
        if i != j:
            wijm_check[i][j] = {}
            for m in range(product_kind):
                wijm_check[i][j][m] = check_model.addVar(lb=0, vtype=GRB.CONTINUOUS,
                                                         name=f'w_{i + 1}_{j + 1}_{m + 1}')
# zij
for i in range(retailer_number):
    zij_check[i] = {}
    for j in range(retailer_number):
        if i != j:
            zij_check[i][j] = check_model.addVar(vtype=GRB.BINARY, name=f'z_{i + 1}_{j + 1}')

beta_check = 0

""" set objective """
check_obj = LinExpr()
for m in range(product_kind):
    for i in range(retailer_number):
        for j in range(retailer_number):
            if i != j:
                check_obj.addTerms(pim[i][m], wijm_check[i][j][m])
for i in range(retailer_number):
    for j in range(retailer_number):
        if i != j:
            check_obj.addTerms(-beta_check * cij_trans[i][j], zij_check[i][j])
for m in range(product_kind):
    for i in range(retailer_number):
        for j in range(retailer_number):
            if i != j:
                check_obj.addTerms(-beta_check * him[i][m], wijm_check[i][j][m])
# for i in range(retailer_number):
#     for m in range(product_kind):
#         for j in range(retailer_number):
#             if i !=j:
#                 check_obj.addTerms(0.5*pim[i][m],wijm_check[j][i][m])
check_model.setObjective(check_obj, GRB.MAXIMIZE)

""" add constraints to subproblem """

# 需求满足率要求约束---DSR0---
# for i in range(retailer_number):
#     for m in range(product_kind):
#         wjim_j_sum = LinExpr()
#         for j in range(retailer_number):
#             if i != j:
#                 wjim_j_sum.addTerms(1, wijm_check[j][i][m])
#                 wjim_j_sum.addTerms(-1, wijm_check[i][j][m])
#         qimk_k_sum = 0
#         for k in range(vehicle_number):
#             qimk_k_sum+= qimk_check[i,m,k]
#         check_model.addConstr(Iim0[i][m] + wjim_j_sum + qimk_k_sum >= 0.01 * dim_real[i][m])

# for i in range(retailer_number):
#     for j in range(retailer_number):
#         if i !=j:
#             for m in range(product_kind):
#                 check_model.addConstr(wijm_check[i][j][m]>=min(ksi_im_check[i,m],miu_im_check[j,m]))

# cons 25-30 ksi definition
for i in range(retailer_number):
    for m in range(product_kind):
        wijm_j_sum = LinExpr()
        wjim_j_sum = LinExpr()
        # zij_j_sum_M = LinExpr()
        for j in range(retailer_number):
            if i != j:
                wijm_j_sum.addTerms(1, wijm_check[i][j][m])
                wjim_j_sum.addTerms(1, wijm_check[j][i][m])
                # zij_j_sum_M.addTerms(ksi_im_check[i, m], zij_check[i][j])
        check_model.addConstr(wijm_j_sum <= ksi_im_check[i, m])
        check_model.addConstr(wjim_j_sum <= Ui[i + 1] * miu_im_check[i, m])
for i in range(retailer_number):
    for j in range(retailer_number):
        if i != j:
            wijm_m_sum = LinExpr()
            ksi_m_sum = 0
            for m in range(product_kind):
                wijm_m_sum.addTerms(1, wijm_check[i][j][m])
                ksi_m_sum += ksi_im_check[i, m]
            check_model.addConstr(wijm_m_sum <= ksi_m_sum * zij_check[i][j])
            # check_model.addConstr(bigM * wijm_m_sum >= zij_check[i][j])

# # cons 32-37 miu definition
# for i in range(retailer_number):
#     for m in range(product_kind):
#         wjim_j_sum = LinExpr()
#         zji_j_sum_M = LinExpr()
#         for j in range(retailer_number):
#             if i != j:
#                 wjim_j_sum.addTerms(1, wijm_check[j][i][m])
#                 zji_j_sum_M.addTerms(miu_im_check[i, m], zij_check[j][i])
#         check_model.addConstr(wjim_j_sum <= zji_j_sum_M,name='32-37')

# cons38-39
for i in range(retailer_number):
    zij_j_sum = LinExpr()
    zji_j_sum = LinExpr()
    for j in range(retailer_number):
        if i != j:
            zij_j_sum.addTerms(1, zij_check[i][j])
            zji_j_sum.addTerms(1, zij_check[j][i])
    ksi_m_sum_M = 0
    miu_m_sum_M = 0
    for m in range(product_kind):
        ksi_m_sum_M += ksi_im_check[i, m]
        miu_m_sum_M += miu_im_check[i, m]
    check_model.addConstr(zij_j_sum <= ksi_m_sum_M)
    check_model.addConstr(zji_j_sum <= miu_m_sum_M)

# cons 40
for i in range(retailer_number):
    Irim_m_sum = 0
    wijm_j_sum = LinExpr()
    wjim_j_sum = LinExpr()
    for m in range(product_kind):
        Irim_m_sum += Irim_check[i, m] * vim[i][m]
        for j in range(retailer_number):
            if i != j:
                wijm_j_sum.addTerms(vim[i][m], wijm_check[i][j][m])
                wjim_j_sum.addTerms(vim[i][m], wijm_check[j][i][m])
    check_model.addConstr(Irim_m_sum - wijm_j_sum + wjim_j_sum <= Ui[i + 1], name=f'<=Ui cons')

# # cons 41
# for i in range(retailer_number):
#     pd_m_sum = 0
#     pwjim_j_sum = LinExpr()
#     pmiu_m_sum = 0
#     for m in range(product_kind):
#         pd_m_sum += (1 - DSR0) * pim[i][m] * dim_real[i][m]
#         pmiu_m_sum += pim[i][m] * miu_im_check[i, m]
#         for j in range(retailer_number):
#             if i != j:
#                 pwjim_j_sum.addTerms(pim[i][m], wijm_check[j][i][m])
#     check_model.addConstr(pd_m_sum + pwjim_j_sum - pmiu_m_sum >= 0,name='DSR0_cons_{}'.format(i))

""" check_model.optimize """
# check_model.write('check_model.lp')
check_model.Params.TimeLimit = 600
check_model.Params.NodefileStart = 0.5
check_model.optimize()
# if (check_model.status != 2):
#     print('infeasible or unbounded! Status: {}'.format(check_model.status))
#     # 计算 IIS
#     check_model.computeIIS()
#     check_model.write("check_model.ilp")
iter_pab = 0
while (check_model.ObjVal > delta_b and iter_pab < max_iter) or (iter_pab == 0):
    # calculate beta
    check_model_numerator = 0
    check_model_denominator = 0
    for m in range(product_kind):
        for i in range(retailer_number):
            for j in range(retailer_number):
                if i != j:
                    check_model_numerator += pim[i][m] * wijm_check[i][j][m].x
    for i in range(retailer_number):
        for j in range(retailer_number):
            if i != j:
                check_model_denominator += cij_trans[i][j] * zij_check[i][j].x
    for m in range(product_kind):
        for i in range(retailer_number):
            for j in range(retailer_number):
                if i != j:
                    check_model_denominator += him[i][m] * wijm_check[i][j][m].x
    if check_model_denominator != 0:
        beta_check = check_model_numerator / check_model_denominator
    else:
        beta_check = 0
    # print('内循环第{}代：(check)beta={}。'.format(iter_pab, beta_check))
    iter_pab += 1

    """ reSet check_model obj """
    check_obj = LinExpr()
    for m in range(product_kind):
        for i in range(retailer_number):
            for j in range(retailer_number):
                if i != j:
                    check_obj.addTerms(pim[i][m], wijm_check[i][j][m])
    for i in range(retailer_number):
        for j in range(retailer_number):
            if i != j:
                check_obj.addTerms(-beta_check * cij_trans[i][j], zij_check[i][j])
    for m in range(product_kind):
        for i in range(retailer_number):
            for j in range(retailer_number):
                if i != j:
                    check_obj.addTerms(-beta_check * him[i][m], wijm_check[i][j][m])
    # for i in range(retailer_number):
    #     for m in range(product_kind):
    #         for j in range(retailer_number):
    #             if i != j:
    #                 check_obj.addTerms(0.5 * pim[i][m], wijm_check[j][i][m])
    check_model.setObjective(check_obj, GRB.MAXIMIZE)
    # solve the resulted sub problem
    stage2_cost = 0
    for i in range(retailer_number):
        for j in range(retailer_number):
            if i != j:
                stage2_cost += cij_trans[i][j] * zij_check[i][j].x
    for m in range(product_kind):
        for i in range(retailer_number):
            for j in range(retailer_number):
                if i != j:
                    stage2_cost += him[i][m] * wijm_check[i][j][m].x

    check_model.optimize()

# print('stage2_cost:')
# print(round(stage2_cost,2))
end_time = time.time()

run_time = end_time - start_time

# print('\n **                check_model Solution             ** \n')
if (check_model.status != 2):
    pass
    # print('The check_model is infeasible or unbounded!')
    # print('infeasible or unbounded! Status: {}'.format(check_model.status))
    # 计算 IIS
    # check_model.computeIIS()
    # check_model.write("check_model.ilp")
else:
    # print('Obj(check_model) : {}'.format(check_model.ObjVal), end='\t | \n')
    # print('final_beta_check={}'.format(beta_check))
    total_stockout_value = 0
    for i in range(retailer_number):
        for m in range(product_kind):
            # if ksi_im_check[i, m] != 0:
            # print(f'(实际需求下转运前情况)节点{i + 1}处商品{m + 1}有余量为：{round(ksi_im_check[i, m],2)}')
            if miu_im_check[i, m] != 0:
                # print(f'(实际需求下转运前情况)节点{i + 1}处商品{m + 1}缺货：{round(miu_im_check[i, m],2)}')
                total_stockout_value += miu_im_check[i, m] * pim[i][m]

    if beta_check != 0:
        check_Z2 = 1 / beta_check
    else:
        check_Z2 = 0
    # print('check_Z2(已还原的 min LR):{}'.format(check_Z2))
    # print(round(check_Z2,3))

    stockout_ratio = {}
    for i in range(retailer_number):
        stock_pim = 0
        demand_pim = 0
        for m in range(product_kind):
            if miu_im_check[i, m] != 0:
                stock_pim += miu_im_check[i, m] * pim[i][m]
                demand_pim += dim_real[i][m] * pim[i][m]
        if demand_pim != 0:
            stockout_ratio[i] = round(stock_pim / demand_pim, 2)
        else:
            stockout_ratio[i] = 0
    #
    # routes = {}
    # for k in range(vehicle_number):
    #     routes[k] = []
    # for i in range(nodes_number):
    #     for j in range(nodes_number):
    #         if i < j:
    #             for k in range(vehicle_number):
    #                 if xijk[i][j][k].x != 0:
    #                     routes[k].append([i, j])
    #                     if xijk[i][j][k].x == 2:
    #                         routes[k].append([j, i])
    #
    # quantities = {}
    # for i in range(retailer_number):
    #     for m in range(product_kind):
    #         for k in range(vehicle_number):
    #             if qimk[i][m][k].x != 0:
    #                 quantities[i, m, k] = round(qimk[i][m][k].x, 0)
    #
    # print('stockout_ratio=\t')
    # print(stockout_ratio)
    # print('routes=\t')
    # print(routes)
    # print('quantities=\t')
    # print(quantities)

    total_stockout_value222 = 0
    for i in range(retailer_number):
        for m in range(product_kind):
            if miu_im_check[i, m] != 0:
                wjim_j = 0
                for j in range(retailer_number):
                    if i != j:
                        wjim_j += wijm_check[j][i][m].x
                if miu_im_check[i, m] - wjim_j > 0:
                    total_stockout_value222 += pim[i][m] * (miu_im_check[i, m] - wjim_j)

    total_stockout_rate = 0
    for i in range(retailer_number):
        for m in range(product_kind):
            if miu_im_check[i, m] != 0:
                wjim_j = 0
                for j in range(retailer_number):
                    if i != j:
                        wjim_j += wijm_check[j][i][m].x
                if miu_im_check[i, m] - wjim_j > 0:
                    if dim_real[i][m] != 0:
                        total_stockout_rate += (miu_im_check[i, m] - wjim_j) / dim_real[i][m]
                    else:
                        total_stockout_rate += 1
    total_stockout_rate = total_stockout_rate / (product_kind * retailer_number)

    total_quantities = 0
    total_pre_demand = 0
    quantities = {}
    for i in range(retailer_number):
        for m in range(product_kind):
            total_pre_demand += dim[i][m]
            for k in range(vehicle_number):
                if qimk[i][m][k].x != 0:
                    quantities[i, m, k] = round(qimk[i][m][k].x, 0)
                    total_quantities += round(qimk[i][m][k].x, 0)

    # for m, var in rm.items():
    #     if var.X != 0:
    #         print(f'rm[{m}]:{var.x}')
    #
    # for i, level1 in xijk.items():
    #     for j, level2 in level1.items():
    #         for k, var in level2.items():
    #             if var.X != 0:
    #                 print(f"xijk_{i}_{j}_{k}: {var.x}")
    # for i,level1 in yik.items():
    #     for k,var in level1.items():
    #         if var.x >smallS:
    #             print(f'yik_{i}_{k}:{var.x}')
    # for i,level1 in Iim.items():
    #     for m,var in level1.items():
    #         if var.x >smallS:
    #             print(f'Iim_{i}_{m}:{var.x}')
    # for (i,m),var in sdim.items():
    #     if var.x >smallS:
    #         print(f'sdim_{i}_{m}:{var.x}')
    # for i, level1 in qimk.items():
    #     for m, level2 in level1.items():
    #         for k, var in level2.items():
    #             if var.X != 0:
    #                 print(f"qimk_{i + 1}_{m}_{k}: {var.x}")
    # for i, level1 in zij_check.items():
    #     for j, var in level1.items():
    #         if var.X != 0:
    #             print(f"zij_{i + 1}_{j + 1}: {var.x}")
    # for i, level1 in wijm_check.items():
    #     for j, level2 in level1.items():
    #         for m, var in level2.items():
    #             if var.X != 0:
    #                 print(f"wijm_{i + 1}_{j + 1}_{m}: {var.x}")
    print("stage1 LR:", round(final_Z1, 5))
    print("stock out:", round(total_stockout_rate, 2))
    print("time:", round(run_time, 2))
    print("total cost:", round(stage1_cost + stage2_cost, 2))
    print("total LR:", round(check_Z2 + final_Z1, 5))
    print("final stock out:", round(total_stockout_value222, 2))
    print("GAP:", Gap)

    # uncertain compare output
    # print("total LR:", round(check_Z2 + final_Z1, 3))
    # print("total cost:", round(stage1_cost + stage2_cost, 2))
    # print("final stock out:", round(total_stockout_value222, 2))
    # print("RAI:", round(total_quantities / total_pre_demand, 2))
    # print("regret:", round(stage2_cost, 2))

    # for m in range(product_kind):
    #     for i in range(retailer_number):
    #         for j in range(retailer_number):
    #             if i != j:
    #                 if wijm_check[i][j][m].x != 0:
    #                     print(f'节点{i + 1}向节点{j + 1}转运商品{m + 1}量为：{round(wijm_check[i][j][m].x,2)}')
