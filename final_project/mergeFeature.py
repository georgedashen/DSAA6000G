# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 10:50:35 2022

@author: 陈焯阳
"""

from pandas.core.indexing import is_label_like
from typing_extensions import final
from config import *
from toolkit import *
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    area_groups = [
        {'id':set([133, 246, 119, 122, 59, 127]),'range':[30,34,98,101]},
        {'id':set([128, 129, 19, 26, 159, 167, 170, 182, 310, 184, 188, 189, 191, 197, 201, 204, 88, 90, 91, 93, 94, 221, 223, 98, 107, 235, 236, 252, 250, 124, 125]),'range':[30,34,101,104]},
        {'id':set([141, 150, 166, 169, 43, 172, 183, 198, 202, 60241, 212, 214, 99, 228, 238, 115, 116, 121, 251]),'range':[30,34,104,107]},
        {'id':set([131, 36, 164, 165, 231, 60139, 174, 175, 206, 303, 82, 51, 243, 55, 308, 119, 313, 318]),'range':[26,30,98,101]},
        {'id':set([256, 130, 132, 147, 148, 149, 151, 153, 32, 33, 35, 60195, 38, 39, 41, 302, 304, 177, 305, 307, 181, 309, 314, 315, 316, 317, 319, 320, 193, 322, 200, 73, 329, 75, 333, 78, 334, 84, 87, 60251, 96, 225, 101, 229, 105, 109, 40047, 240, 247, 120, 254, 255]),'range':[26,30,101,104]},
        {'id':set([352, 321, 355, 324, 326, 328, 331, 77, 47, 48, 335, 339]),'range':[26,30,104,107]},
        {'id':set([161, 226, 137, 138, 171, 140, 113, 306, 152, 186, 220, 60157]),'range':[22,26,98,101]},
        {'id':set([50117, 327, 106, 332, 142, 146, 24, 155, 156, 29]),'range':[22,26,101,104]}
    ]

    for i,area in enumerate(area_groups):
        for flag in ['train', 'valid']:
            _magn_res = pd.read_csv(f"./area_feature/area_{i}_{flag}_magn.csv")
            _magn_res.drop(['Unnamed: 0', 'Day_day_max_mean',
            'Day_day_min_mean',
            'Day_day_mean_max',
            'Day_day_mean_min',
            'Day_mean',
            'Day_max',
            'Day_min',
            'Day_max_min',
            'Day_lastday_mean',
            'Day_lastday_max',
            'Day_lastday_min',
            'Day_lastday_max_min','label_M','label_long','label_lati'],axis=1,inplace=True)
            
            for name in _magn_res.columns.to_list()[1:]:
                _magn_res.rename(columns={name:(name+'_magn')},inplace=True)
            _magn_res.columns = _magn_res.columns.str.replace('.*@', '', regex=True)
            
            _sound_res = pd.read_csv(f"./area_feature/area_{i}_{flag}_sound.csv")
            _sound_res.drop('Unnamed: 0', axis=1, inplace=True)
            for name in _sound_res.columns.to_list()[1:-3]:
                _sound_res.rename(columns={name:(name+'_sound')},inplace=True)
            _sound_res.columns = _sound_res.columns.str.replace('.*@', '', regex=True)
            
            _final_res = pd.merge(_magn_res,_sound_res,on='Day',how='left')
            _final_res.dropna(inplace=True)
            _final_res.to_csv(f'./area_feature/area_{i}_{flag}.csv')