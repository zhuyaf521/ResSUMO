import os.path
import re
import sys
import time
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.keras import layers, optimizers
import pandas as pd
from tensorflow.keras.models import Model
from Bio import SeqIO


def file_check(filename):
    if not os.path.exists(filename):
        print("file does not exist")
        sys.exit()
    else:
        with open(filename, "r") as handle:
            fasta = SeqIO.parse(handle, "fasta")
            return any(fasta)


def get_dataset(filepath):
    try:
        predict_id = []
        predict_seq = []
        for index, record in enumerate(SeqIO.parse(filepath, 'fasta')):
            re_search = re.search(r"\|[-A-Za-z0-9]+\|", record.name)
            if re_search:
                name = re_search.group()[1:-1]
            else:
                name = record.name
            sequences = 'X' * 19 + str(record.seq) + 'X' * 19
            for location, seq in enumerate(sequences):
                if seq == 'K':
                    predict_id.append(name + '*' + str(location + 1 - 19))
                    predict_seq.append(sequences[location - 19:location + 20])
        csvfile = pd.DataFrame({'Protein': predict_id, 'Sequence': predict_seq})

        return csvfile
    except:
        return pd.DataFrame()


def ZScale(dataframe):
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

    return np.array(encodings).astype(np.float64)


def res_net_block(input_data, filters, strides=1):
    x = layers.Conv1D(filters, kernel_size=3, strides=strides, padding='same')(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv1D(filters, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    if strides != 1:
        down_sample = layers.Conv1D(filters, kernel_size=1, strides=strides)(input_data)
    else:  # 否就直接连接
        down_sample = input_data
    x = layers.Add()([x, down_sample])
    output = layers.Activation('relu')(x)
    return output


def ResSUMOModel(feature):
    inputs1 = tf.keras.Input(shape=(feature.shape[1], feature.shape[2]))
    x = layers.Conv1D(128, kernel_size=1)(inputs1)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPool1D(pool_size=2, strides=1, padding='same')(x)
    x = layers.Dropout(0.5)(x)

    x = res_net_block(x, 128)
    x = layers.MaxPool1D(2)(x)
    x = layers.Dropout(0.5)(x)

    x = res_net_block(x, 128)
    x = layers.MaxPool1D(2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

    output = layers.Dense(1, activation=tf.nn.sigmoid)(x)
    ResSUMO = Model(inputs=[inputs1], outputs=[output], name="DeepSUMO")
    ResSUMO.compile(optimizer=optimizers.Adam(),
                    loss='binary_crossentropy',
                    metrics=['accuracy'],
                    experimental_run_tf_function=False)
    return ResSUMO


def predict(dataframe, model_path, save_path):
    x = ZScale(dataframe)
    sign = list(dataframe['Protein'])
    name = []
    position = []
    for s in sign:
        reversal = s[::-1]
        site = reversal.index('*')
        name.append(reversal[site + 1:][::-1])
        position.append(reversal[:site][::-1])
    folds = [1, 2, 3, 4, 5]
    predict_score = np.zeros((len(x), len(folds)))
    predict_result = []
    predict_confidence = []
    for fold in folds:
        modelName = 'model' + str(fold) + '.h5'
        modelPath = os.path.join(model_path, modelName)
        network = ResSUMOModel(x)
        network.load_weights(modelPath)
        predict_score[:, fold - 1:fold] = network.predict(x)

    predict_average_score = np.average(predict_score, axis=1)
    predict_average_score = np.around(predict_average_score, 3)
    for i in predict_average_score:
        if i >= 0.85:
            predict_result.append(1)
            predict_confidence.append('Very High confidence')
        elif i >= 0.7:
            predict_result.append(2)
            predict_confidence.append('High confidence')
        elif i >= 0.5:
            predict_result.append(3)
            predict_confidence.append('Medium confidence')
        else:
            predict_result.append(0)
            predict_confidence.append('No')
    saveCsv = pd.DataFrame({'Protein': name, 'Position': position, 'Sequence': dataframe['Sequence'],
                            'Prediction score': predict_average_score, 'Prediction category': predict_confidence})
    saveCsv.to_csv(save_path, index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="it's usage tip.",
                                     description="ResSUMO: A Deep Learning Architecture Based on Residual Structure "
                                                 "for Lysine SUMOylation Sites Prediction")
    parser.add_argument("--file", required=True, help="input fasta format file")

    parent_dir = os.path.abspath(os.path.dirname(os.getcwd()))
    args = parser.parse_args()
    filecheck = file_check(args.file)
    if filecheck:
        dataset = get_dataset(args.file)
        net_path = os.path.join(parent_dir, 'Models')
        res_path = os.path.join(parent_dir, 'Results')
        result_path = os.path.join(res_path, str(time.time()).split('.')[0] + '.csv')
        predict(dataset, net_path, result_path)
    else:
        print("The input file format is incorrect, it must be in fasta format")
        sys.exit()
    print("The prediction results are stored in ", result_path)
