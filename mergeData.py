##some station is not available and concat all usable file, total 50 stationID in training but only 44 is OK
##use tqdm to add process bar

import os
import pandas as pd
import re
from tqdm import tqdm #process bar
from config import * #select proper window and step, features
from toolkit import * #some self defined functinos

if __name__ == "__main__":
    Data_Folder_Path = '../training/'
    fileName_list = os.listdir(Data_Folder_Path)
    stationInfo_list = pd.read_csv(Station_Info_Path)

    #筛选可用台站：选取同时有地声、电磁数据且在eqlst.csv内标记为未来可用的台站
    _continueable_stations = stationInfo_list[stationInfo_list['MagnUpdate']&stationInfo_list['SoundUpdate']]['StationID'].unique()
    _continueable_stations = set(_continueable_stations)
    re_magn = re.compile(r'(\d+)_magn.csv')
    re_sound = re.compile(r'(\d+)_sound.csv')
    _set_magn = set()
    _set_sound = set()
    for filename in fileName_list:
        _magn_match = re_magn.findall(filename)
        _sound_match = re_sound.findall(filename)
        if(_magn_match):
            _set_magn.add(int(_magn_match[0]))
            continue
        if(_sound_match):
            _set_sound.add(int(_sound_match[0]))
            continue
    usable_stations = _continueable_stations&_set_magn&_set_sound
    dump_object(Usable_Station_Path, usable_stations)

    print('合并数据:')
    for type in ('magn', 'sound'):
        res = []
        for _id in tqdm(usable_stations, desc=f'{type}:'):
            _df = pd.read_csv(Data_Folder_Path+str(_id)+f'_{type}.csv')[Used_features[type]]
            res.append(_df)
        final_df = pd.concat(res)
        final_df.to_pickle(Merged_Data_Path[type])
        del(final_df)
    
    