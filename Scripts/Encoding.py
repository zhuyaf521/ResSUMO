import pandas as pd
import os,sys,re
import numpy as np
import itertools
from collections import Counter
from numpy import *

def Binary(filepath, CodeType):
    AA = 'ACDEFGHIKLMNPQRSTVWYX'
    encodings = []
    dataframe = pd.read_csv(filepath)
    label = list(dataframe['Label'])
    sequences = list(dataframe['Sequence'])
    for i in sequences:
        sequence = re.sub('[^ACDEFGHIKLMNPQRSTVWYX]', 'X', ''.join(i).upper())
        code = []
        for aa in sequence:
            singlecode = []
            for aa1 in AA:
                tag = 1 if aa == aa1 else 0
                singlecode.append(tag)
            code.append(singlecode)
        encodings.append(code)
    if CodeType == 1:
        return np.array(encodings).astype(np.float64), np.array(label).astype(np.float64)
    else:
        return np.array(encodings).astype(np.float64).reshape(len(encodings),-1), np.array(label).astype(np.float64)
    
def EAAC(filepath, CodeType, windows=5):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    dataframe = pd.read_csv(filepath)
    label = list(dataframe['Label'])
    sequences = list(dataframe['Sequence'])
    for i in sequences:
        sequence = re.sub('[^ACDEFGHIKLMNPQRSTVWYX]', 'X', ''.join(i).upper())
        code = []
        for j in range(len(sequence)):
            singlecode = []
            if j < len(sequence) and j + windows <= len(sequence):
                count = Counter(re.sub('X', '', sequence[j:j + windows]))
                for key in count:
                    count[key] = count[key] / windows
                for aa in AA:
                    singlecode.append(count[aa])
                code.append(singlecode)
        encodings.append(code)
    if CodeType == 1:
        return np.array(encodings).astype(np.float64), np.array(label).astype(np.float64)
    else:
        return np.array(encodings).astype(np.float64).reshape(len(encodings),-1), np.array(label).astype(np.float64)

def Binary_of_bigram(filepath):
    aaPairs = []
    encodings = []
    for i in itertools.product('ACDEFGHIKLMNPQRSTVWY', repeat=2):
        aaPairs.append(''.join(i))
    dataframe = pd.read_csv(filepath)
    label = list(dataframe['Label'])
    sequences = list(dataframe['Sequence'])
    for i in sequences:
        sequence = re.sub('[^ACDEFGHIKLMNPQRSTVWYX]', 'X', ''.join(i).upper())
        code = []
        for index in range(len(sequence)-1):
            singercode = np.zeros(len(aaPairs))
            pattern = '' + sequence[index] + sequence[index+1]
            if pattern in aaPairs:
            	    singercode[aaPairs.index(pattern)] = 1
            code.append(singercode)
        encodings.append(code)
    return np.array(encodings).astype(np.float64), np.array(label).astype(np.float64)
    
def BLOSUM62(filepath,CodeType):
    blosum62 = {
        'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],  # A
        'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
        'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],  # N
        'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],  # D
        'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
        'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],  # Q
        'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],  # E
        'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],  # G
        'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],  # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],  # L
        'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],  # K
        'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],  # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],  # P
        'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],  # S
        'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],  # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],  # Y
        'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],  # V
        'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # X
    }
    encodings = []
    dataframe = pd.read_csv(filepath)
    label = list(dataframe['Label'])
    sequences = list(dataframe['Sequence'])
    for i in sequences:
        sequence = re.sub('[^ACDEFGHIKLMNPQRSTVWYX]', 'X', ''.join(i).upper())
        code = []
        for aa in sequence:
            singlecode = []
            if aa in blosum62.keys():
                singlecode = singlecode + blosum62[aa]
            else:
                singlecode = singlecode + blosum62['-']
            code.append(singlecode)
        encodings.append(code)
    if CodeType == 1:
        return np.array(encodings).astype(np.float64), np.array(label).astype(np.float64)
    else:
        return np.array(encodings).astype(np.float64).reshape(len(encodings),-1), np.array(label).astype(np.float64)
    
def ZScale(filepath, CodeType):
    zscale = {
        'A': [0.24, -2.32, 0.60, -0.14, 1.30],  # A
        'C': [0.84, -1.67, 3.71, 0.18, -2.65],  # C
        'D': [3.98, 0.93, 1.93, -2.46, 0.75],  # D
        'E': [3.11, 0.26, -0.11, -0.34, -0.25],  # E
        'F': [-4.22, 1.94, 1.06, 0.54, -0.62],  # F
        'G': [2.05, -4.06, 0.36, -0.82, -0.38],  # G
        'H': [2.47, 1.95, 0.26, 3.90, 0.09],  # H
        'I': [-3.89, -1.73, -1.71, -0.84, 0.26],  # I
        'K': [2.29, 0.89, -2.49, 1.49, 0.31],  # K
        'L': [-4.28, -1.30, -1.49, -0.72, 0.84],  # L
        'M': [-2.85, -0.22, 0.47, 1.94, -0.98],  # M
        'N': [3.05, 1.62, 1.04, -1.15, 1.61],  # N
        'P': [-1.66, 0.27, 1.84, 0.70, 2.00],  # P
        'Q': [1.75, 0.50, -1.44, -1.34, 0.66],  # Q
        'R': [3.52, 2.50, -3.50, 1.99, -0.17],  # R
        'S': [2.39, -1.07, 1.15, -1.39, 0.67],  # S
        'T': [0.75, -2.18, -1.12, -1.46, -0.40],  # T
        'V': [-2.59, -2.64, -1.54, -0.85, -0.02],  # V
        'W': [-4.36, 3.94, 0.59, 3.44, -1.59],  # W
        'Y': [-2.54, 2.44, 0.43, 0.04, -1.47],  # Y
        'X': [0.00, 0.00, 0.00, 0.00, 0.00],  # X
    }
    encodings = []
    dataframe = pd.read_csv(filepath)
    label = list(dataframe['Label'])
    sequences = list(dataframe['Sequence'])
    for i in sequences:
        sequence = re.sub('[^ACDEFGHIKLMNPQRSTVWYX]', 'X', ''.join(i).upper())
        code = []
        for aa in sequence:
            singlecode = []
            if aa in zscale.keys():
                singlecode = singlecode + zscale[aa]
            else:
                singlecode = singlecode + zscale['-']
            code.append(singlecode)
        encodings.append(code)
    if CodeType == 1:
        return np.array(encodings).astype(np.float64), np.array(label).astype(np.float64)
    else:
        return np.array(encodings).astype(np.float64).reshape(len(encodings),-1), np.array(label).astype(np.float64)
    
def EGAAC(filepath, CodeType, window = 5):
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }
    groupKey = group.keys()
    encodings = []
    dataframe = pd.read_csv(filepath)
    label = list(dataframe['Label'])
    sequences = list(dataframe['Sequence'])
    for i in sequences:
        sequence = re.sub('[^ACDEFGHIKLMNPQRSTVWYX]', 'X', ''.join(i).upper())
        code = []
        for j in range(len(sequence)):
            if j + window <= len(sequence):
                singlecode = []
                count = Counter(re.sub('X', '', sequence[j:j + window]))
                myDict = {}
                for key in groupKey:
                    for aa in group[key]:
                        myDict[key] = myDict.get(key, 0) + count[aa]
                for key in groupKey:
                    singlecode.append(myDict[key] / window)
                code.append(singlecode)
        encodings.append(code)
    if CodeType == 1:
        return np.array(encodings).astype(np.float64), np.array(label).astype(np.float64)
    else:
        return np.array(encodings).astype(np.float64).reshape(len(encodings),-1), np.array(label).astype(np.float64)
    
def getPSSM(filepath, CodeType):
    encodings = []
    label = []
    files = os.listdir(filepath) # 读入文件夹
    sorted_files = sorted(files,key = lambda i:int(re.findall(r'_(\d+)-',i)[0]))
    for file in sorted_files:
        label.append(int(file[file.find('.')-1]))
        pssm = []
        with open(os.path.join(filepath, file)) as f:
            lines = f.readlines()[3:38]
            pssm = np.array([line.split()[2:22] for line in lines], dtype=int)
            f.close()
            encodings.append(pssm)
    if CodeType == 1:
        return np.array(encodings).astype(np.float64), np.array(label).astype(np.float64)
    else:
        return np.array(encodings).astype(np.float64).reshape(len(encodings),-1), np.array(label).astype(np.float64)

    
def AAindex(filepath, CodeType):
    AA = 'ARNDCQEGHILKMFPSTWYV'
    fileAAindex = r'D:\jupyter\DeepSUMO-master\Utils\data\Sort_AAindex.txt'
    with open(fileAAindex) as f:
        records = f.readlines()[1:15]
    AAindex = []
    AAindexName = []
    for i in records:
        AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
        AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)
    encodings = []
    dataframe = pd.read_csv(filepath)
    label = list(dataframe['Label'])
    sequences = list(dataframe['Sequence'])
    for i in sequences:
        sequence = re.sub('[^ACDEFGHIKLMNPQRSTVWYX]', 'X', ''.join(i).upper())
        code = []
        for j in sequence:
            single_code = []
            for k in AAindexName:
                if j in AA:
                    value = AAindex[AAindexName.index(k)][AA.index(j)]
                else:
                    value = 0
                single_code.append(value)
            code.append(single_code)
        encodings.append(code)
    if CodeType == 1:
        return np.array(encodings).astype(np.float64), np.array(label).astype(np.float64)
    else:
        return np.array(encodings).astype(np.float64).reshape(len(encodings),-1), np.array(label).astype(np.float64)
        
def CKSAAP(filepath, gap = 0):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    aaPairs = []
    patten = re.compile('X|U|-|_')
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)
    encodings = []
    dataframe = pd.read_csv(filepath)
    label = list(dataframe['Label'])
    sequences = list(dataframe['Sequence'])
    for i in sequences:
        sequence = re.sub('[^ACDEFGHIKLMNPQRSTVWYX]', 'X', ''.join(i).upper())
        code = []
        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                    index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sum = sum + 1
            for pair in aaPairs:
                code.append(myDict[pair] / sum)
        encodings.append(code)
    return np.array(encodings).astype(np.float64), np.array(label).astype(np.float64)
    
# Construct four sets of features (0, 1 notation) based on the properties of amino acids that appear at -2, -1 and +1, +2
def Statistics_property(filepath):
    dataframe = pd.read_csv(filepath)
    label = list(dataframe['Label'])
    sequences = list(dataframe['Sequence'])
    encodings = []
    middle = len(sequences[0])//2
    positive_cahrge = 'DE'
    negative_charge = 'RKH'
    charge = 'DERKH'
    Hydrophobic = 'AFGILPVWY'
    type1 = 'ILV'
    type2 = 'AFMPW'
    type3 = 'GY'
    
    for i in sequences:
        sequence = re.sub('[^ACDEFGHIKLMNPQRSTVWYX]', 'X', ''.join(i).upper())
        code = zeros(11)
        if sequence[middle-2] in positive_cahrge:
            code[0] = 1
        else:
            if sequence[middle-2] in negative_charge:
                code[1] = 1
        if sequence[middle-1] in type1:
            code[5] = 1
        else:
            if sequence[middle-1] in type2:
                code[4] = 1
            else:
                if sequence[middle-1] in type3:
                    code[3] = 1
                else:
                    code[2] = 1
        if sequence[middle-1] in charge:
            code[6] = 1
        if sequence[middle+1] in Hydrophobic:
            code[7] = 1
        if sequence[middle+1] in charge:
            code[8] = 1
        if sequence[middle+2] in positive_cahrge:
            code[9] = 1
        else:
            if sequence[middle+2] in negative_charge:
                code[10] = 1
        
        encodings.append(code)
    return np.array(encodings).astype(np.float64), np.array(label).astype(np.float64)
    
def getSequence(train_fastas):
    pos_list = []
    neg_list = []
    sequences = list(train_fastas['Sequence'])
    labels = list(train_fastas['Label'])
    for s, l in zip(sequences, labels):
        if l==1:
            pos_list.append(s)
        else:
            neg_list.append(s)
    return pos_list, neg_list

# Determine whether it is a natural amino acid, use 'X' to represent all other amino acids, and construct a frequency matrix
def replace_no_native_amino_acid(lists):
    native_amino_acid = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',)
    frequency_array = zeros((21, len(lists[0])))
    flag = 1
    for site in range(len(lists)):
        for i in range(len(lists[0])):
            for j in range(len(native_amino_acid)):
                if lists[site][i] == native_amino_acid[j]:
                    frequency_array[j][i] = frequency_array[j][i] + 1
                    flag = 0
                    break
            if flag != 0:
                frequency_array[20][i] = frequency_array[20][i]+1
            flag = 1

    length = len(lists)
    for i in range(len(frequency_array)):
        for j in range(len(frequency_array[0])):
            frequency_array[i][j] =frequency_array[i][j]/length
    
    return frequency_array

# Subtract the negative sample frequency from the positive sample frequency to get the overall sample frequency
def result_frequency_matrix(positive_matrix, negative_matrix):
    result_matrix = zeros((len(positive_matrix), len(positive_matrix[0])))
    for i in range(len(positive_matrix)):
        for j in range(len(positive_matrix[0])):
            result_matrix[i][j] = positive_matrix[i][j] - negative_matrix[i][j]
    return result_matrix


# Convert ordinal representation to frequency representation or entropy representation
def to_site(lists, frequency_array):
    full_frequency_array = []
    for i in range(len(lists)):
        position = int(lists[i])
        full_frequency_array.append(frequency_array[position-1][i])
    return full_frequency_array

#Convert amino acid sequence to ordinal representation
def number_encoding(peptide):
    native_amino_acid = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',)
    numlists = zeros(len(peptide))
    flag = 1
    for i in range(len(peptide)):
        for j in range(len(native_amino_acid)):
            if peptide[i] == native_amino_acid[j]:
                numlists[i] = j+1
                flag = 0
                break
        if flag != 0:
            numlists[i] = 21
        flag = 1
    
    return numlists

def PSAAP(trainpath, testpath = None):
    traindf = pd.read_csv(trainpath)
    
    psiteList, nsiteList = getSequence(traindf)
    positive_frequency_matrix = replace_no_native_amino_acid(psiteList)
    negative_frequency_matrix = replace_no_native_amino_acid(nsiteList)
    frequency_matrix = result_frequency_matrix(positive_frequency_matrix, negative_frequency_matrix)
    
    train_encodings = []
    train_label = list(traindf['Label'])
    sequences = list(traindf['Sequence'])
    for i in sequences:
        sequence = re.sub('[^ACDEFGHIKLMNPQRSTVWYX]', 'X', ''.join(i).upper())
        number_seq = number_encoding(sequence)
        code = to_site(number_seq, frequency_matrix)
        train_encodings.append(code)
    if testpath:
        testdf = pd.read_csv(testpath)
        test_encodings = []
        test_label = list(testdf['Label'])
        sequences1 = list(testdf['Sequence'])
        for i in sequences1:
            sequence = re.sub('[^ACDEFGHIKLMNPQRSTVWYX]', 'X', ''.join(i).upper())
            number_seq = number_encoding(sequence)
            code = to_site(number_seq, frequency_matrix)
            test_encodings.append(code)
        return np.array(train_encodings).astype(np.float64), np.array(train_label).astype(np.float64), np.array(test_encodings).astype(np.float64), np.array(test_label).astype(np.float64)
    
    return np.array(train_encodings).astype(np.float64), np.array(train_label).astype(np.float64)