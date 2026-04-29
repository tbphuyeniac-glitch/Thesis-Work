from gurobipy import *
import numpy as np
import pandas as pd
import re
import math
import time
import sys
from revise_data import *

start_time = time.time()

np.random.seed(17)

""" The input parameter of stage 1"""

# retailer_number, product_kind, vehicle_number, Ui,cij, pim,Qk,bk,vim,him,Iim0,dim_real,dim \
#     = parse_data_file(sys.argv[1])
# data_dic = parse_data_file(sys.argv[1])
data_dic= parse_data_file('./uncertain_compare-0.0/mirplr-5-9-5-0.09-5.dat')
retailer_number, product_kind, vehicle_number, Ui,cij, pim,Qk,bk,vim,him,Iim0,dim_real,dim \
    =data_dic["num_customers"],data_dic["num_products"],data_dic["num_vehicles"],data_dic["Ui"],data_dic["cij"],data_dic["pim"], \
data_dic["Qk"],data_dic["bk"],data_dic["vim"],data_dic["him"],data_dic["Iim0"],data_dic["dim_actual"],data_dic["dim_predict"]
# print("retailer_number:", retailer_number)
# print("product_kind:", product_kind)
# print("vehicle_number:", vehicle_number)
# print("Ui:", Ui)
# print("pim:", pim)
# print("Qk:", Qk)
# print("bk:", bk)
# print("vim:", vim)
# print("him:", him)
# print("Iim0:", Iim0)
# print("dim_real:", dim_real)
# print('dim:',dim)



# retailer_number, product_kind, vehicle_number, Ui, x_co, y_co, dim_real, pim, Qk, bk, vim, him, Iim0 \
#     = gdf.update_data(sys.argv[1])
# volume_coef=float(sys.argv[2])
# price_coef=float(sys.argv[3])
# vim=[[element * volume_coef for element in row] for row in vim]
# pim=[[element * price_coef for element in row] for row in pim]
# coefficient=float(sys.argv[2])
# dim=gdf.generate_predemand(dim_real,sigma=float(sys.argv[2]))
# dim_real = gdf.generate_predemand(dim)
# cij = gdf.euclidean_distance(x_co, y_co)

# cij = [[element for element in row] for row in cij]
nodes_number = retailer_number + 1
""" The input parameter of stage 2"""
coefficient = 1.2
cij_trans = [[element * coefficient for element in row] for row in cij]

# tau_T = retailer_number * product_kind
tau_T = retailer_number * product_kind
DSR0 = 0.9
bigM = 999999
smallS=0.00001
gamma = 1
""" build initial master problem """
""" Create variables """
master = Model('master problem')

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
        sdim[i, m] = master.addVar(lb=0, ub=dim_real[i][m], vtype=GRB.CONTINUOUS, name=f'sdim_{i + 1}_{m + 1}')
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

# 参数算法中的参数设置
iter_cnt = 0
alpha = {}
alpha[iter_cnt] = 0

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

master.setObjective(master_obj, GRB.MAXIMIZE)

""" Add Constraints  """
# cons 2
for m in range(product_kind):
    qimk_ik_sum = LinExpr()
    for k in range(vehicle_number):
        for i in range(retailer_number):
            qimk_ik_sum.addTerms(1, qimk[i][m][k])
    master.addConstr(Iim[0][m] - (Iim0[0][m] + rm[m] - qimk_ik_sum)<=smallS, name=f'cons(2)-m{m + 1}_1')
    master.addConstr(Iim[0][m] - (Iim0[0][m] + rm[m] - qimk_ik_sum) <= -smallS, name=f'cons(2)-m{m + 1}_2')

# cons 3
for i in range(retailer_number):
    for m in range(product_kind):
        # print(f'dim[{i}][{m}]:{dim[i][m]}')
        # print(f'real_dim[{i}][{m}]:{dim_real[i][m]}')
        # master.addConstr(sdim[i, m] >=1* dim[i][m])
        qimk_k_sum = LinExpr()
        for k in range(vehicle_number):
            qimk_k_sum.addTerms(1, qimk[i][m][k])
            # master.addConstr(qimk[i][m][k] <= dim[i][m])
        master.addConstr(Iim[i + 1][m] - (Iim0[i + 1][m] + qimk_k_sum - sdim[i, m])<= smallS, name=f'cons(3)-i{i + 1}-m{m + 1}_1')
        master.addConstr(Iim[i + 1][m] - (Iim0[i + 1][m] + qimk_k_sum - sdim[i, m])>= -smallS, name=f'cons(3)-i{i + 1}-m{m + 1}_2')

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
        for k in range(vehicle_number):
            qimk_km_sum.addTerms(vim[i][m], qimk[i][m][k])
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

""" solve the model and output  """
# master.Params.TimeLimit=200
# master.Params.NodefileStart=0.5
master.optimize()
# master.write('dbmaster.lp')
""" Column-and-constraint generation """
max_iter = 10
delta = 0.001
delta_a = 0.001
delta_b = 0.001
# print('\n **                Initial Solution             ** \n')
#
# if the model is infeasible，calculate IIS
if master.Status != 2:
    print("stage1_Model is infeasible:{}".format(master.Status))
    # calculate IIS
    master.computeIIS()
    master.write("deterministic_model_IIS.ilp")
# else:
#     print('*'*100+'\n(master)Obj = {}'.format(master.objVal))
#     for k in range(vehicle_number):
#         for i in range(retailer_number):
#             for m in range(product_kind):
#                     if qimk[i][m][k].x != 0:
#                         print(f'第{k + 1}辆车向节点{i+1}配送商品{m+1}量：{qimk[i][m][k].x}')

# close the outputflag
master.setParam('Outputflag', 0)

"""
 Main loop of parameter algorithm 
"""

master.optimize()
# master.write("deterministic_model_IIS.ilp")

while ((master.ObjVal) > delta_a and iter_cnt < max_iter) or (iter_cnt == 0):
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
    iter_cnt += 1
    if master_denominator == 0:
        alpha[iter_cnt] = 0
    else:
        alpha[iter_cnt] = master_numerator / master_denominator

    # print('内循环第{}代：alpha={}'.format(iter_cnt,alpha[iter_cnt]))

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
    master.setObjective(master_obj, GRB.MAXIMIZE)
    master.optimize()

# print('Obj(stage1): {}'.format(master.ObjVal), end='\t |')
# for k in range(vehicle_number):
#     for i in range(retailer_number):
#         for m in range(product_kind):
#             if qimk[i][m][k].x != 0:
#                 print(f'第{k + 1}辆车向节点{i+1}配送商品{m + 1}量：{qimk[i][m][k].x}')

# master.write('finalMP.lp')
# print('\n\nOptimal solution found !')
# print('Opt_Obj_alpha : {}'.format(alpha[iter_cnt]))
if alpha[iter_cnt] != 0:
    final_Z1 = 1 / alpha[iter_cnt]
else:
    final_Z1 = 0
# print('final_Z1(已还原的 min LR):{}'.format(final_Z1))
# print('stage1 LR:')
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


# uncertain compare output
# print("opt total:", round(stage1_cost, 2))
print("stage1_LR: ",round(final_Z1, 5))

