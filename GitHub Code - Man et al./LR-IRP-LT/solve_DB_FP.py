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
data_dic = parse_data_file(sys.argv[1])
# data_dic= parse_data_file('./instance_BFP_remain/mirplr-15-1-3-2.dat')
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


# print("solve")
""" solve the model and output  """
master.Params.TimeLimit=200
master.Params.NodefileStart=0.5

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
    # master.computeIIS()
    # master.write("deterministic_model_IIS.ilp")
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
# print('total cost (without qimk)')
# print(stage1_cost)

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
#                 print(f'第{k + 1}辆车：向节点{i+1}配送商品{m + 1}量：{qimk[i][m][k].x}')
# for m in range(product_kind):
#     print(f'配送中心进购产品{m + 1}量：{rm[m].x}')


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

check_model.setObjective(check_obj, GRB.MAXIMIZE)

""" add constraints to subproblem """
# for i in range(retailer_number):
#     for j in range(retailer_number):
#         if i !=j:
#             for m in range(product_kind):
#                 check_model.addConstr(wijm_check[i][j][m]>=min(ksi_im_check[i,m],miu_im_check[j,m]))

# cons 20-21
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
        check_model.addConstr(wjim_j_sum <= Ui[i+1] * miu_im_check[i, m])
# cons 18-19
for i in range(retailer_number):
    for j in range(retailer_number):
        if i != j:
            wijm_m_sum = LinExpr()
            ksi_m_sum=0
            for m in range(product_kind):
                wijm_m_sum.addTerms(1, wijm_check[i][j][m])
                ksi_m_sum+=ksi_im_check[i, m]
            check_model.addConstr(wijm_m_sum <= ksi_m_sum * zij_check[i][j])
            # check_model.addConstr(bigM*wijm_m_sum>=zij_check[i][j])

# # cons 32-37 miu definition
# for i in range(retailer_number):
#     for m in range(product_kind):
#         wjim_j_sum = LinExpr()
#         zji_j_sum_M = LinExpr()
#         for j in range(retailer_number):
#             if i != j:
#                 wjim_j_sum.addTerms(1, wijm_check[j][i][m])
#                 zji_j_sum_M.addTerms(miu_im_check[i, m], zij_check[j][i])
#         check_model.addConstr(wjim_j_sum <= zji_j_sum_M)

# cons21-22
for i in range(retailer_number):
    zij_j_sum = LinExpr()
    zji_j_sum = LinExpr()
    for j in range(retailer_number):
        if i != j:
            zij_j_sum.addTerms(1, zij_check[i][j])
            zji_j_sum.addTerms(1, zij_check[j][i])
    ksi_m_sum = 0
    miu_m_sum = 0
    for m in range(product_kind):
        ksi_m_sum += ksi_im_check[i, m]
        miu_m_sum += miu_im_check[i, m]
    check_model.addConstr(zij_j_sum <= ksi_m_sum)
    check_model.addConstr(zji_j_sum <= miu_m_sum)

# cons 23
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
    check_model.addConstr(Irim_m_sum - wijm_j_sum + wjim_j_sum <= Ui[i + 1])

# cons 41
# for i in range(retailer_number):
#     pd_m_sum = 0
#     pwjim_j_sum = LinExpr()
#     pmiu_m_sum = 0
#     for m in range(product_kind):
#         pd_m_sum+=(1 - DSR0) * pim[i][m] *dim_real[i][m]
#         pmiu_m_sum+=pim[i][m]*miu_im_check[i,m]
#         for j in range(retailer_number):
#             if i != j:
#                 pwjim_j_sum.addTerms(pim[i][m], wijm_check[j][i][m])
#     check_model.addConstr(pd_m_sum + pwjim_j_sum - pmiu_m_sum >= 0, name='DSR0_cons_{}'.format(i))

""" check_model.optimize """
# check_model.write('check_model.lp')
check_model.Params.TimeLimit=200
check_model.Params.NodefileStart=0.5
check_model.optimize()
# print('(循环前的)此时的DSR0={}'.format(DSR0))
# print('check_model.Status:{}'.format(check_model.Status))
# while check_model.Status !=2:
#     if DSR0>0:
#         DSR0-=0.05
#     else:
#         # print("DSR0已不为正数！")
#         break
#     # update DSR0cons
#     for i in range(retailer_number):
#         check_model.remove(check_model.getConstrByName('DSR0_cons_{}'.format(i)))
#         pd_m_sum = 0
#         pwjim_j_sum = LinExpr()
#         pmiu_m_sum = 0
#         for m in range(product_kind):
#             pd_m_sum += (1 - DSR0) * pim[i][m] * dim_real[i][m]
#             pmiu_m_sum += pim[i][m] * miu_im_check[i, m]
#             for j in range(retailer_number):
#                 if i != j:
#                     pwjim_j_sum.addTerms(pim[i][m], wijm_check[j][i][m])
#         check_model.addConstr(pd_m_sum + pwjim_j_sum - pmiu_m_sum >= 0, name='DSR0_cons_{}'.format(i))
#         check_model.optimize()

if (check_model.status != 2):
    print('The check_model is infeasible or unbounded!Status: {}'.format(check_model.status))
    # print('Status: {}'.format(check_model.status))
    # 计算 IIS
    # check_model.computeIIS()
    # check_model.write("check_model.ilp")

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
# print(stage2_cost)
# for i in range(retailer_number):
#     for m in range(product_kind):
#         for k in range(vehicle_number):
#             if qimk[i][m][k].x != 0:
#                 print(i, m, k)
#                 print(round(qimk[i][m][k].x,0))
#                 print(f'dim:{dim[i][m]}')

# print('\n **                check_model Solution             ** \n')
if (check_model.status != 2):
    # print('The check_model is infeasible or unbounded!')
    print('infeasible or unbounded! Status: {}'.format(check_model.status))
    # 计算 IIS
    # check_model.computeIIS()
    # check_model.write("check_model.ilp")
else:
    # print('Obj(check_model) : {}'.format(check_model.ObjVal), end='\t | \n')
    # print('final_beta_check={}'.format(beta_check))

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

    routes = {}
    for k in range(vehicle_number):
        routes[k] = []
    for i in range(nodes_number):
        for j in range(nodes_number):
            if i < j:
                for k in range(vehicle_number):
                    if xijk[i][j][k].x != 0:
                        routes[k].append([i, j])
                        if xijk[i][j][k].x == 2:
                            routes[k].append([j, i])
    total_quantities=0
    total_pre_demand=0
    quantities = {}
    for i in range(retailer_number):
        for m in range(product_kind):
            total_pre_demand+=dim[i][m]
            for k in range(vehicle_number):
                if qimk[i][m][k].x != 0:
                    quantities[i, m, k] = round(qimk[i][m][k].x, 0)
                    total_quantities+=round(qimk[i][m][k].x, 0)

    # print('stockout_ratio=')
    # print(stockout_ratio)
    # print('routes=')
    # print(routes)
    # print('quantities=')
    # print(quantities)

    total_stockout_value = 0
    for i in range(retailer_number):
        for m in range(product_kind):
            # if ksi_im_check[i, m] != 0:
            # print(f'(实际需求下转运前情况)节点{i + 1}处商品{m + 1}有余量为：{round(ksi_im_check[i, m],2)}')
            if miu_im_check[i, m] != 0:
                # print(f'(实际需求下转运前情况)节点{i + 1}处商品{m + 1}缺货：{round(miu_im_check[i, m],2)}')
                total_stockout_value += miu_im_check[i, m] * pim[i][m]
    # print(round(total_stockout_value,2))

    if beta_check != 0:
        check_Z2 = 1 / beta_check
    else:
        check_Z2 = 0
    # print('check_Z2(已还原的 min LR):{}'.format(check_Z2))
    # print(round(check_Z2,3))

    total_stockout_value222=0
    for i in range(retailer_number):
        for m in range(product_kind):
            if miu_im_check[i, m] != 0:
                wjim_j=0
                for j in range(retailer_number):
                    if i !=j:
                        wjim_j+=wijm_check[j][i][m].x
                if miu_im_check[i, m]-wjim_j>0:
                    total_stockout_value222+=pim[i][m]*(miu_im_check[i, m]-wjim_j)

    total_stockout_rate = 0
    for i in range(retailer_number):
        for m in range(product_kind):
            # if ksi_im_check[i, m] !=0:
            #     print(f"ksi_im_check[{i}, {m}]:{ksi_im_check[i, m]}")
            if miu_im_check[i, m] != 0:
                # print(f"miu_im_check[{i}, {m}]:{miu_im_check[i, m]}")
                wjim_j = 0
                for j in range(retailer_number):
                    if i != j:
                        wjim_j += wijm_check[j][i][m].x
                if miu_im_check[i, m] - wjim_j > 0:
                    if dim_real[i][m] != 0:
                        total_stockout_rate += (miu_im_check[i, m] - wjim_j) / dim_real[i][m]
                        # print((miu_im_check[i, m] - wjim_j) / dim_real[i][m])
                    else:
                        total_stockout_rate += 1
    total_stockout_rate = total_stockout_rate / (product_kind*retailer_number)


    for i in range(retailer_number):
        for j in range(retailer_number):
            if i != j:
                wijm_m_sum = 0
                for m in range(product_kind):
                    wijm_m_sum+=wijm_check[i][j][m].x
                if zij_check[i][j].x==1:
                    pass
                    # print(f"z_{i}_{j}:{zij_check[i][j].x}")
                    # print(f"w_{i}_{j}_m_sum:{wijm_m_sum}")


    # for m, var in rm.items():
    #     if var.X != 0:
    #         print(f'rm[{m}]:{var.x}')
    #
    # for i, level1 in xijk.items():
    #     for j, level2 in level1.items():
    #         for k, var in level2.items():
    #             if var.X >smallS:
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
    #             if var.X >smallS:
    #                 print(f"qimk_{i+1}_{m}_{k}: {var.x}")
    # for i, level1 in zij_check.items():
    #     for j, var in level1.items():
    #         if var.X >smallS:
    #             print(f"zij_{i+1}_{j+1}: {var.x}")
    # for i, level1 in wijm_check.items():
    #     for j, level2 in level1.items():
    #         for m, var in level2.items():
    #             if var.X >smallS:
    #                 print(f"wijm_{i+1}_{j+1}_{m}: {var.x}")

    # for m in range(product_kind):
    #     for i in range(retailer_number):
    #         for j in range(retailer_number):
    #             if i != j:
    #                 if wijm_check[i][j][m].x != 0:
    #                     print(f'节点{i+1}向节点{j+1}转运商品{m + 1}量为：{wijm_check[i][j][m].x}')
    # for i in range(retailer_number):
    #     for m in range(product_kind):
    #         if ksi_im_check[i,m] != 0:
    #             print(f'(实际)节点{i+1}处商品{m + 1}有余量为：{ksi_im_check[i,m]}')
    #         if miu_im_check[i,m] != 0:
    #             print(f'(实际)节点{i+1}处商品{m + 1}缺货：{miu_im_check[i,m]}')

end_time = time.time()

run_time = end_time - start_time

#  experiment output
print("stage1 LR:", round(final_Z1, 5))
print("stock out:", round(total_stockout_rate, 2))
print("time:", round(run_time, 2))
print("total cost:", round(stage1_cost + stage2_cost, 2))
print("total LR:", round(check_Z2 + final_Z1, 5))
print("final stock out:", round(total_stockout_value222, 2))

# uncertain compare output
# print("total LR:", round(check_Z2 + final_Z1, 3))
# print("total cost:", round(stage1_cost + stage2_cost, 2))
# print("final stock out:", round(total_stockout_value222, 2))
# print("RAI:",round(total_quantities/total_pre_demand,2))
# print("regret:",round(stage2_cost,2))
