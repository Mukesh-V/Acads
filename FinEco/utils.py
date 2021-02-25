import pandas as pd
import matplotlib.pyplot as plt

def readCSV(file, invert=False):
	df = pd.read_csv(file, header=None)
	if invert:
		return df.T.iloc[1:,:], df.T.iloc[0, :]
	return df, df.iloc[0,:]

def split(data, ratio):
	pt = int(ratio * data.shape[0])
	train_df = data.iloc[:pt, :]
	test_df  = data.iloc[pt:, :]

	data_dict = {
		'train': xy(train_df),
		'test' : xy(test_df)
	}
	return data_dict

def xy(data):
	last = data.shape[1]
	y = data[last]
	x = data.drop(columns=last)
	return {'x' : x, 'y': y}

def plot(fdata, y, op):
	year = 2020.0
	qtrs = []
	n = 4
	for i in range(n):
		for j in range(4):
			qtrs.append(year - i - j/4.0)

	title = fdata['name'] + ' - ' + fdata['sect']
	image = fdata['dir'] + fdata['name'] + '.png'
	fig = plt.figure()
	
	fig.suptitle(title, fontsize=15)
	plt.grid(True)
	plt.xlabel('Quarters', fontsize=12)
	plt.ylabel('Opening price (in Rs.)', fontsize=12)

	y = y.astype(float)
	act, = plt.plot(qtrs, y,  c='black')
	pre, = plt.plot(qtrs, op, c='red')
	plt.legend([act, pre],['Actual', 'Predicted'])
	plt.savefig(image)
	plt.close()
