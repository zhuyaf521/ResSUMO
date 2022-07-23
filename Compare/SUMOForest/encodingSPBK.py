import docx
import pandas as pd
from numpy import *
import numpy as np
import math

def getSite(filename):
    doc = docx.Document(filename)
    siteList = []
    for site in doc.paragraphs:
        siteList.append(site.text)
    #print(siteList)
    return siteList

# 判断是否为天然氨基酸，用X代表其他所有氨基酸，所有氨基酸用数字来替代，构造单个频率矩阵以及n_gram频率矩阵
def replace_no_native_amino_acid(lists, datatype):
    native_amino_acid = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',)
    numlists = zeros((len(lists), 22))
    frequency_array = zeros((21, 21))
    n_gram_frequency_array = zeros((441, 20))
    flag = 1
    for site in range(len(lists)):
        for i in range(len(lists[0])):# 位点位置
            for j in range(len(native_amino_acid)):# 氨基酸种类
                if lists[site][i] == native_amino_acid[j]:
                    numlists[site][i] = j+1
                    flag = 0
                    if i > 0 & i < (len(lists[0])-2):
                        a = (numlists[site][i-1]-1) * 20
                        b = numlists[site][i] - 1
                        n_gram_index = int(a + b)
                        n_gram_frequency_array[n_gram_index][i-1] = n_gram_frequency_array[n_gram_index][i-1] + 1
                    break
            if flag != 0:
                numlists[site][i] = 21
            flag = 1
            if datatype == 1:
                numlists[site][21] = 1
    length = len(lists)
    for i in range(len(n_gram_frequency_array)):
        for j in range(len(n_gram_frequency_array[0])):
            n_gram_frequency_array[i][j] = n_gram_frequency_array[i][j]/length
    return n_gram_frequency_array

# 构造skip_gram的频率矩阵
def get_skip_gram_frequency_array(lists, datatype, k):
    native_amino_acid = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',)
    numlists = zeros((len(lists), 22))
    skip_gram_frequency_array = zeros((441, 20-k))
    flag = 1
    for site in range(len(lists)):
        for i in range(len(lists[0])):  # 位点位置
            for j in range(len(native_amino_acid)):  # 氨基酸种类
                if lists[site][i] == native_amino_acid[j]:
                    numlists[site][i] = j + 1
                    flag = 0
                    if i > k:
                        a = (numlists[site][i - k - 1] - 1) * 20
                        b = numlists[site][i] - 1
                        skip_gram_index = int(a + b)
                        skip_gram_frequency_array[skip_gram_index][i - k - 1] = skip_gram_frequency_array[skip_gram_index][i-k-1] + 1
                    break
            if flag != 0:
                numlists[site][i] = 21
            flag = 1
            if datatype == 1:
                numlists[site][21] = 1
    length = len(lists)
    for i in range(len(skip_gram_frequency_array)):
        for j in range(len(skip_gram_frequency_array[0])):
            skip_gram_frequency_array[i][j] = skip_gram_frequency_array[i][j] / length
    return skip_gram_frequency_array


# 正样本频率减去负样本频率得到整体样本频率
def result_frequency_site(positive_site, negative_site,datatype):
    result_site = zeros((len(positive_site), len(positive_site[0])))
    for i in range(len(positive_site)):
        for j in range(len(positive_site[0])):
            if datatype == "frequency":
                result_site[i][j] = positive_site[i][j] - negative_site[i][j]
            elif datatype == "entropy":
                result_site[i][j] = negative_site[i][j] - positive_site[i][j]
    return result_site

# 特征矩阵拼接
def splice_feature_array(feature_array_x, feature_array_y):
    sum_feature_array = zeros((len(feature_array_x), len(feature_array_x[0])+len(feature_array_y[0])-1))
    for site in range(len(sum_feature_array)):
        for i in range(len(feature_array_x[0])-1):
            sum_feature_array[site][i] = feature_array_x[site][i]
        for i in range(len(feature_array_x[0])-1, len(sum_feature_array[0])):
            sum_feature_array[site][i] = feature_array_y[site][i+1-len(feature_array_x[0])]
    return sum_feature_array

# 把序号表示矩阵转换到n_gram频率表示
def to_n_gram_site(lists, datatype, n_gram_frequency_array):
    full_n_gram_frequency_array = zeros((len(lists), 21))
    # print(len(lists))
    for site in range(len(lists)):
        for i in range(len(lists[0])-1):#位点位置
            if i > 0:
                position_a = int(lists[site][i-1]) - 1
                position_b = int(lists[site][i]) - 1
                position_index = int(position_a * 20 + position_b)
                full_n_gram_frequency_array[site][i-1] = n_gram_frequency_array[position_index][i-1]
        if datatype == 1:#标记正负样本
            full_n_gram_frequency_array[site][20] = 1
            
    return full_n_gram_frequency_array

# 把序号表示矩阵转换到skip_gram频率表示
def to_skip_gram_site(lists, datatype,skip_gram_frequency_array, k):
    full_skip_gram_frequency_array = zeros((len(lists), 21-k))
    # print(len(lists))
    for site in range(len(lists)):
        for i in range(len(lists[0])-1):#位点位置
            if i > k:
                position_a = int(lists[site][i-1-k]) - 1
                position_b = int(lists[site][i]) - 1
                position_index = int(position_a * 20 + position_b)
                full_skip_gram_frequency_array[site][i-1-k] = skip_gram_frequency_array[position_index][i-1-k]
        if datatype == 1:#标记正负样本
            arraylen = len(full_skip_gram_frequency_array[0])-1
            full_skip_gram_frequency_array[site][arraylen] = 1

    return full_skip_gram_frequency_array

# 保证生成的随机状态一致
random_state = 2018
np.random.seed(random_state)

# 获取氨基酸的序号表示
def get_numlists(lists, datatype):
    native_amino_acid = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',)
    numlists = zeros((len(lists), 22))
    flag = 1
    for site in range(len(lists)):
        for i in range(len(lists[0])):# 位点位置
            for j in range(len(native_amino_acid)):# 氨基酸种类
                if lists[site][i] == native_amino_acid[j]:
                    numlists[site][i] = j+1
                    flag = 0
                    break
            if flag != 0:
                numlists[site][i] = 21
            flag = 1
            if datatype == 1:
                numlists[site][21] = 1

    return numlists

# 根据氨基酸在—1和+2的出现的氨基酸的特性构造两组特征（0，1表示法）
def hydrophobic_position_array0(allarrary, datatype):
    native_amino_acid = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',)
    type1 = ('I', 'L', 'V',)
    # 0.4388
    type2 = ('A', 'F', 'M', 'P', 'W',)
    # -0.031
    type3 = ('G', 'Y',)
    # -0.0644
    # other：-0.3725
    type4 = ('D', 'E',)
    # 0.6287
    # -0.6299
    position1_array = zeros((len(allarrary), 6))
    for i in range(len(allarrary)-1):

        # print(allarrary[i][9])
        if allarrary[i][9] in type1:
            for j in range(4):
                if j == 3:
                    position1_array[i][j] = 1
                else:
                    position1_array[i][j] = 0
        elif allarrary[i][9] in type2:
            for j in range(4):
                if j == 2:
                    position1_array[i][j] = 1
                else:
                    position1_array[i][j] = 0
        elif allarrary[i][9] in type3:
            for j in range(4):
                if j == 1:
                    position1_array[i][j] = 1
                else:
                    position1_array[i][j] = 0
        else:
            for j in range(4):
                if j == 0:
                    position1_array[i][j] = 1
                else:
                    position1_array[i][j] = 0

        # print(allarrary[i][12])
        if allarrary[i][12] in type4:
            position1_array[i][4] = 0
        else:
            position1_array[i][4] = 1
        position1_array[i][5] = datatype
    # for r in position1_array:
    #     print(r)
    return position1_array

def get_BKencoding(positive_array, negative_array, sum_n_gram_frequency_site, sum_skip_gram_frequency_site1, sum_skip_gram_frequency_site2):
    positive = 1
    negative = 0
    positive_n_gram_frequency_array = to_n_gram_site(positive_array, positive, sum_n_gram_frequency_site)
    negative_n_gram_frequency_array = to_n_gram_site(negative_array, negative, sum_n_gram_frequency_site)
    n_gram_frequency_allarray = np.concatenate((positive_n_gram_frequency_array, negative_n_gram_frequency_array), axis=0)
    
    positive_skip_gram_frequency_array1 = to_skip_gram_site(positive_array, positive, sum_skip_gram_frequency_site1, 1)
    negative_skip_gram_frequency_array1 = to_skip_gram_site(negative_array, negative, sum_skip_gram_frequency_site1, 1)
    skip_gram_frequency_allarray1 = np.concatenate(
        (positive_skip_gram_frequency_array1, negative_skip_gram_frequency_array1), axis=0)
    
    positive_skip_gram_frequency_array2 = to_skip_gram_site(positive_array, positive, sum_skip_gram_frequency_site2, 2)
    negative_skip_gram_frequency_array2 = to_skip_gram_site(negative_array, negative, sum_skip_gram_frequency_site2, 2)
    skip_gram_frequency_allarray2 = np.concatenate(
        (positive_skip_gram_frequency_array2, negative_skip_gram_frequency_array2), axis=0)
    
    min_skip_gram_frequency_allarray=splice_feature_array(skip_gram_frequency_allarray1,skip_gram_frequency_allarray2)
    feature_array = splice_feature_array(n_gram_frequency_allarray, min_skip_gram_frequency_allarray)
    
    return feature_array

def get_Sequence(p_file, n_file, flag, windows=41):
    if flag:
        psiteList = getSite(p_file)
        nsiteList = getSite(n_file)
    else:
        window = windows//2
        psiteList = []
        nsiteList = []
        E_pos = pd.read_csv(p_file)
        E_neg = pd.read_csv(n_file)
        for m in list(E_pos['Sequence']):
            psiteList.append(m[window-10:window+10+1])
        for n in list(E_neg['Sequence']):
            nsiteList.append(n[window-10:window+10+1])
    return psiteList, nsiteList
       
def get_Matrix(psiteList, nsiteList):
    
    positive = 1
    negative = 0
    positive_n_gram_frequency_site = replace_no_native_amino_acid(psiteList, positive)
    negative_n_gram_frequency_site = replace_no_native_amino_acid(nsiteList, negative)
    # n_gram频率矩阵
    sum_n_gram_frequency_site = result_frequency_site(positive_n_gram_frequency_site, negative_n_gram_frequency_site, "frequency")

    # skip_gram频率矩阵 1，2，3
    positive_skip_gram_frequency_site1 = get_skip_gram_frequency_array(psiteList, positive, 1)
    negative_skip_gram_frequency_site1 = get_skip_gram_frequency_array(nsiteList, negative, 1)
    sum_skip_gram_frequency_site1 = result_frequency_site(positive_skip_gram_frequency_site1,
                                                         negative_skip_gram_frequency_site1, "frequency")

    positive_skip_gram_frequency_site2 = get_skip_gram_frequency_array(psiteList, positive, 2)
    negative_skip_gram_frequency_site2 = get_skip_gram_frequency_array(nsiteList, negative, 2)
    sum_skip_gram_frequency_site2 = result_frequency_site(positive_skip_gram_frequency_site2,
                                                         negative_skip_gram_frequency_site2, "frequency")
    
    return sum_n_gram_frequency_site, sum_skip_gram_frequency_site1, sum_skip_gram_frequency_site2