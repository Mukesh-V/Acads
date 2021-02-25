import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from utils import *
from model import Regressor 

cwd = os.getcwd()
data_dir = cwd + "/data/"
plot_dir = cwd + "/plots/"
file_names = os.listdir(data_dir)
files = [data_dir + x for x in file_names]

scores = []
coeffs = []
model = Regressor()
for i in range(len(files)):
	if files[i][-3:] == 'csv':
		data, headers = readCSV(files[i], True)
		data = data.drop(columns=0)
		data_dict = split(data, ratio=0.9)

		model.inject(data_dict)
		model.run(file_names[i][:-4])
		score = model.validate()

		scores.append({
			'Stock': file_names[i][:-4],
			   'R2': score['r2'],
			 'MAPE': score['mape']
		})
		coeffs.append(np.append(model.regressor.coef_, model.regressor.intercept_).tolist())
		
		name = file_names[i][:-4]
		sector = headers[1:].tolist()[-2]
		fdata = {
			'dir' : plot_dir,
			'name': name,
			'sect': sector
		}

		whole = xy(data)
		op = model.regressor.predict(whole['x'])
		plot(fdata, whole['y'], op)

scores_df = pd.DataFrame.from_dict(scores)
scores_df.to_csv('scores.csv', index=False)

cols = headers[1:].tolist()
cols[-2] = 'Nifty Sector'
cols[-1] = 'Intercept'
coeffs_df = pd.DataFrame(coeffs, columns=cols)
coeffs_df.to_csv('coeffs.csv', index=False)
