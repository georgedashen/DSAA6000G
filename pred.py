import lightgbm as lgb
from numpy.lib.function_base import extract
import pandas as pd
import re
import numpy as np
import os
import zipfile
from toolkit import *
import warnings
warnings.filterwarnings('ignore')

re_train = re.compile(r'area_\d+_train.csv')
re_valid = re.compile(r'area_\d+_valid.csv')

_token = '9632b4a24ae849d68839d5d7632d358b'

binary_model = 'precision_0_Eval100'
magnitude_model = 'weightedPrecisionEval1000'


def download_aeta_data_to_file(data_type, date_range_str, save_to, token, oversea=False):
	assert data_type in ('EM', 'GA', 'EM&GA')
	import requests
	oversea = 'true' if oversea else 'false'
	url = 'https://api.competition.aeta.cn/downloadByToken'
	params = {
		'dataType': data_type,
		'date': date_range_str,
		'oversea': oversea,
	}
	resp = requests.get(url, params=params, headers={'Authorization': token})
	resp.raise_for_status()
	with open(save_to, 'wb') as fd:
		for chunk in resp.iter_content(chunk_size=128):
			fd.write(chunk)

def cacu_features_inf(df:pd.DataFrame, window:int, step:int,tag:str,Time_Range) ->pd.DataFrame:
	'''
	
	'''
	if(len(df)==0):
		return None
	df.reset_index(drop=True,inplace=True)
	averageName = tag+'@abs_mean'
	df.rename(columns={averageName:'average'},inplace=True)
	df['average'] = df['average']-df['average'].mean()
	df['diff_1'] = df.groupby('StationID')['average'].shift(1)
	df['diff_1'] = df['average'].values - df['diff_1'].values
	
	df.loc[:,'Day'] = df['TimeStamp']
	_start_timestamp = string2stamp(Time_Range[0])
	df['Day'] = df['Day'] - _start_timestamp
	df['Day'] = (df['Day']//86400+1).astype(int)
	df.reset_index()
	tmp = pd.DataFrame(sorted(df['Day'].unique()))
	tmp.columns=['Day']
	res_df = pd.DataFrame((tmp['Day']//step+1).unique()).astype(int)
	res_df.columns=['Day']
	res_df['Day'] = res_df['Day']*step
	
	for feature in df.columns.to_list()[2:]:
		if(feature=='Day'):continue
		for tagging in ['max', 'min', 'mean']:
			kk = df.groupby('Day')[feature].agg(tagging)
			kk.rename(f'{feature}_day_{tagging}',inplace=True)
			tmp = pd.merge(tmp, kk, how='left',on='Day')
		#max_mean.min_mean:
		tmp[f'{feature}_day_max_mean'] = tmp[f'{feature}_day_max'].rolling(window=window,center=False).mean()
		tmp[f'{feature}_day_min_mean'] = tmp[f'{feature}_day_min'].rolling(window=window,center=False).mean()
		#mean_max,mean_min:
		tmp[f'{feature}_day_mean_max'] = tmp[f'{feature}_day_mean'].rolling(window=window,center=False).max()
		tmp[f'{feature}_day_mean_min'] = tmp[f'{feature}_day_mean'].rolling(window=window,center=False).min()
		res_df = pd.merge(res_df,tmp[['Day',f'{feature}_day_max_mean',f'{feature}_day_min_mean',f'{feature}_day_mean_max',f'{feature}_day_mean_min']],on='Day',how='left')
	
		res_df[f'{feature}_mean'] = None
		res_df[f'{feature}_max'] = None
		res_df[f'{feature}_min'] = None
		res_df[f'{feature}_max_min'] = None
		for i,row in res_df.iterrows():
			endDay = row['Day']
			startDay = endDay - window
			data_se = df[(df['Day']>startDay)&(df['Day']<=endDay)][feature]
			res_df[f'{feature}_mean'].iloc[i] = data_se.mean()
			res_df[f'{feature}_max'].iloc[i] = data_se.max()
			res_df[f'{feature}_min'].iloc[i] = data_se.min()
			res_df[f'{feature}_max_min'].iloc[i] = data_se.max() - data_se.min()
	
		res_df[f'{feature}_lastday_mean'] = None
		res_df[f'{feature}_lastday_max'] = None
		res_df[f'{feature}_lastday_min'] = None
		res_df[f'{feature}_lastday_max_min'] = None
		for i,row in res_df.iterrows():
			endDay = row['Day']
			data_last = df[df['Day']==endDay][feature]
			res_df[f'{feature}_lastday_mean'].iloc[i] = data_last.mean()
			res_df[f'{feature}_lastday_max'].iloc[i] = data_last.max()
			res_df[f'{feature}_lastday_min'].iloc[i] = data_last.min()
			res_df[f'{feature}_lastday_max_min'].iloc[i] = data_last.max() - data_last.min()
	
	for name in res_df.columns.to_list():
		if(name=='Day'):continue
		res_df.rename(columns={name:(name+'_'+tag)},inplace=True)
	res_df.dropna(axis=0,how='any',inplace=True)
	res_df.reset_index(drop=True,inplace=True)
	return res_df

if __name__ == "__main__":
	save_path = '../test_week/'
	time_range = ['20220515', '20220521']
	#time_range = ['20220529', '20220604']
	file_name = time_range[0]+'-'+time_range[1]
	extractpath = '../test_week/'+file_name+'/'
	em_path = f'{extractpath}EM_{time_range[0]}-{time_range[1]}/'
	ga_path = f'{extractpath}GA_{time_range[0]}-{time_range[1]}/'
	if(len(os.listdir(extractpath)) != 2):
		download_aeta_data_to_file("EM&GA",file_name,save_path+file_name+'.zip', _token)
		frzip = zipfile.ZipFile(save_path+file_name+'.zip','r')
		extractfile = frzip.namelist()
		frzip.extractall(extractpath)
		frzip.close()
		# em_path = f'{extractpath}EM&GA_{time_range[0]}-{time_range[1]}/EM_{time_range[0]}-{time_range[1]}/'
		# ga_path = f'{extractpath}EM&GA_{time_range[0]}-{time_range[1]}/GA_{time_range[0]}-{time_range[1]}/'	
		for filename in os.listdir(em_path):
			if(filename.endswith('.csv')):continue
			with zipfile.ZipFile(em_path+filename,'r') as frzip:
				frzip.extractall(em_path)
		for filename in os.listdir(ga_path):
			if(filename.endswith('.csv')):continue
			with zipfile.ZipFile(ga_path+filename,'r') as frzip:
				frzip.extractall(ga_path)

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

	max_mag = -1
	eq_area = -1
	for area in (0, 1, 2, 3, 4, 5, 6, 7):
		sid = area_groups[area]['id']

		em_list = []
		for id_num in sid:
			try:
				em_list.append(pd.read_csv(em_path+f'{id_num}_magn.csv'))
			except:
				continue
		
		if(len(em_list)==0): continue
		em_data = pd.concat(em_list)
		del em_list
		ga_list = []
		for id_num in sid:
			try:
				ga_list.append(pd.read_csv(ga_path+f'{id_num}_sound.csv'))
			except:
				continue
		if(len(ga_list)==0): continue
		ga_data = pd.concat(ga_list)
		del ga_list
		
		em_data = cacu_features_inf(em_data, 7, 7,'magn',time_range)
		ga_data = cacu_features_inf(ga_data, 7, 7,'sound',time_range)
		for name in em_data.columns.to_list()[1:]:
			em_data.rename(columns={name:(name+'_magn')},inplace=True)
		em_data.columns = em_data.columns.str.replace('.*@', '', regex=True)
		for name in ga_data.columns.to_list()[1:]:
			ga_data.rename(columns={name:(name+'_sound')},inplace=True)
		ga_data.columns = ga_data.columns.str.replace('.*@', '', regex=True)
		
		_final_res = pd.merge(em_data,ga_data,on='Day',how='left')
		_final_res.fillna(0,inplace=True)
		_final_res.drop('Day',axis=1,inplace=True)
		features = _final_res.iloc[-1]

		Bin = lgb.Booster(model_file=f'./model/{area}_binary_hp_{binary_model}_model.txt')
		clf = lgb.Booster(model_file=f'./model/{area}_mag_{magnitude_model}_model.txt')
		
		result_bin = Bin.predict(features, num_iteration=Bin.best_iteration)
		result_bin = pd.Series([1 if i > 0.5 else 0 for i in result_bin])
		result_mag = np.matrix(clf.predict(features, num_iteration=clf.best_iteration))
		result_mag = pd.Series(np.array(result_mag.argmax(axis=1)).reshape(1,))
		print(f'Area_{area}: mag{result_mag[0]}, bin{result_bin[0]}')
		result_combined = result_mag
		for i, ss in enumerate(result_bin):
			if(ss==1):
				result_combined[i] = 0

		if result_combined[0] == 0:
			continue
		else:
			if(result_combined[0] > max_mag):
				max_mag = result_combined[0]
				eq_area = area
	
	magn_level = {0:0, 1:3.7, 2:4.2, 3:4.7, 4:5}
	long = (area_groups[eq_area]['range'][2]+area_groups[eq_area]['range'][3])/2
	lati = (area_groups[eq_area]['range'][0]+area_groups[eq_area]['range'][1])/2
	print(f'震级:{magn_level[max_mag]}, 震中:经度{long}, 纬度{lati}')


