#路径参数

Data_Folder_Path = './data/'                      #原始数据文件夹
Eq_list_path = Data_Folder_Path+'eqlst.csv'               #地震目录文件
Station_Info_Path = Data_Folder_Path+'StationInfo.csv'      #台站信息文件
Usable_Station_Path = Data_Folder_Path + 'UsableStation.bin'    #可用台站集

Used_features = {'magn':['StationID',
 'TimeStamp',
 'magn@var',
 'magn@power',
 'magn@skew',
 'magn@kurt',
 'magn@abs_max',
 'magn@abs_mean',
 'magn@abs_max_top5p',
 'magn@abs_max_top10p',
 'magn@energy_sstd',
 'magn@energy_smax',
 'magn@power_0_5',
 'magn@power_5_10',
 'magn@power_10_15',
 'magn@power_15_20',
 'magn@power_20_25',
 'magn@power_25_30',
 'magn@power_30_35',
 'magn@power_35_40',
 'magn@power_40_60',
 'magn@power_140_160',
 'magn@power_other',
 'magn@frequency_center',
 'magn@mean_square_frequency',
 'magn@variance_frequency',
 'magn@frequency_entropy',
 'magn@level4d_abs_mean',
 'magn@level4d_energy',
 'magn@level4d_energy_smax',
 'magn@level4d_energy_sstd',
 'magn@level5d_abs_mean',
 'magn@level5d_energy',
 'magn@level5d_energy_smax',
 'magn@level5d_energy_sstd',
 'magn@level6a_abs_mean',
 'magn@level6a_energy',
 'magn@level6a_energy_smax',
 'magn@level6a_energy_sstd',
 'magn@level6d_abs_mean',
 'magn@level6d_energy',
 'magn@level6d_energy_smax',
 'magn@level6d_energy_sstd',
 'magn@ulf_abs_mean',
 'magn@ulf_var',
 'magn@ulf_power',
 'magn@ulf_skew',
 'magn@ulf_kurt',
 'magn@ulf_abs_max',
 'magn@ulf_abs_max_top5p',
 'magn@ulf_abs_max_top10p',
 'magn@ulf_energy_sstd',
 'magn@ulf_energy_smax'],
  'sound':['StationID',
   'TimeStamp',
   'sound@var',
   'sound@power',
   'sound@skew',
   'sound@kurt',
   'sound@abs_max',
   'sound@abs_mean',
   'sound@abs_max_top5p',
   'sound@abs_max_top10p',
   'sound@energy_sstd',
   'sound@energy_smax',
   'sound@s_zero_rate',
   'sound@s_zero_rate_max',
   'sound@power_0_5',
   'sound@power_5_10',
   'sound@power_10_15',
   'sound@power_15_20',
   'sound@power_20_25',
   'sound@power_25_30',
   'sound@power_30_35',
   'sound@power_35_40',
   'sound@power_40_60',
   'sound@power_140_160',
   'sound@power_other',
   'sound@frequency_center',
   'sound@mean_square_frequency',
   'sound@variance_frequency',
   'sound@frequency_entropy',
   'sound@level4d_abs_mean',
   'sound@level4d_energy',
   'sound@level4d_energy_smax',
   'sound@level4d_energy_sstd',
   'sound@level5d_abs_mean',
   'sound@level5d_energy',
   'sound@level5d_energy_smax',
   'sound@level5d_energy_sstd',
   'sound@level6a_abs_mean',
   'sound@level6a_energy',
   'sound@level6a_energy_smax',
   'sound@level6a_energy_sstd',
   'sound@level6d_abs_mean',
   'sound@level6d_energy',
   'sound@level6d_energy_smax',
   'sound@level6d_energy_sstd',
   'sound@mean']}#选取电磁、地声的绝对值均值特征

#合并数据文件
Merged_Data_Path= {'magn':Data_Folder_Path + 'magn_data.pkl',
                    'sound':Data_Folder_Path + 'sound_data.pkl'}

#训练集与验证集生成
Time_Range = {'train':['20161001','20200331'],              #训练集时间段
                'valid':['20200401','20201231']}                #验证集时间段                        
Window = 7                                                      #窗长，单位 天
Step = 7                                                        #步长，单位 天