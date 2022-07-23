# -*- coding: utf-8 -*-
"""
描述:
    Goldstein-Price函数测试，其控制变量为开区间，因此可设一个较大范围进行寻优搜索
    该函数是经典的二元多项式函数模型，由A.A.Goldstein和J.F.Price于1971年首次提出，
    用于研究局部最小化算法，

Created on Tue Oct 23 22:43:02 2018

@author: TRsky
"""

import numpy as np
import geatpy as ga

# 注意：不建议把目标函数放在执行脚本内，建议放在另一个文件中
def aimfuc(Phen, LegV): # 定义目标函数
    x = Phen[:, [0]]
    y = Phen[:, [1]]
    f = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 13 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y **2))
    return [f, LegV] # 注意返回的参数要符合Geatpy数据结构

if __name__ == "__main__":
    AIM_M = __import__('sga_test_Goldstein') # 获取函数所在文件的地址
    # 变量设置
    ranges = np.array([[-2, -100], [100, 2]])  # 生成自变量的范围矩阵
    borders = np.array([[1, 1], [1, 1]])   # 生成自变量的边界矩阵（1表示变量的区间是闭区间）
    precisions = [1, 1] # 根据crtfld的函数特性，这里需要设置精度为任意正值，否则在生成区域描述器时会默认为整数编码，并对变量范围作出一定调整
    FieldDR = ga.crtfld(ranges, borders, precisions) # 生成区域描述器
    # 调用编程模板
    [pop_trace, var_trace, times] = ga.sga_new_real_templet(AIM_M, 'aimfuc', None, None, FieldDR, problem = 'R', maxormin = 1, MAXGEN = 200, NIND = 50, SUBPOP = 1, GGAP = 0.9, selectStyle = 'sus', recombinStyle = 'xovdp', recopt = None, pm = None, distribute = False, drawing = 1)
