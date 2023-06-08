# -*- coding: utf-8 -*-
"""
@author lizheng
@date  10:23
@packageName
@className cal_linear_index
@software PyCharm
@version 1.0.0
@describe TODO
"""
import os
from coefficient_calu import corr_value
import numpy as np
import pandas as pd


def run(idx):
    root_path = r'E:\homegate\R_Quality_Assessment\Dataset\G-PCD\subjective scores\desktop setup'
    type_list = ['bunny', 'cube', 'dragon', 'sphere', 'vase']
    mos = []
    for d in ['D01', 'D02']:
        df = pd.read_csv(os.path.join(root_path, d, 'subj_desktop_dsis.csv'))
        for i in range(df.__len__()):
            if i % 5 == 0:
                continue
            mos.append(df.loc[i, 'MOS'])

    with open('../tmp/scores_for_test_gpcd_%d.txt' % idx, 'r') as f:
        f_str = f.read().rstrip('\n')
    pred = [float(i) for i in f_str.split('\n')]

    mos = np.array(mos, dtype=float)
    pred = np.array(pred, dtype=float)
    plcc, srcc, krcc, rmse = corr_value(mos, pred)
    return plcc, srcc, krcc, rmse


if __name__ == '__main__':
    plcc_list, srcc_list, krcc_list, rmse_list = [], [], [], []
    for i in range(20):
        plcc, srcc, krcc, rmse = run(i)
        print('scores_for_test_gpcd_%d.txt' % i)
        print('plcc: ', plcc)
        plcc_list.append(plcc)
        print('srcc: ', srcc)
        srcc_list.append(srcc)
        print('krcc: ', krcc)
        krcc_list.append(krcc)
        print('rmse: ', rmse)
        rmse_list.append(rmse)
        print('\n')

    print('plcc mean: %f, max: %f' % (np.mean(plcc_list), np.max(plcc_list)))
    print('srcc mean: %f, max: %f' % (np.mean(srcc_list), np.max(srcc_list)))
    print('krcc mean: %f, max: %f' % (np.mean(krcc_list), np.max(krcc_list)))
    print('rmse mean: %f, max: %f' % (np.mean(rmse_list), np.max(rmse_list)))
