from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt 

class Regressor:
	def __init__(self):
		self.regressor = LinearRegression(fit_intercept=True)
		self.train = None
		self.test  = None

		print('Regressor Initiated')

	def inject(self, data):
		self.train = data['train']
		self.test  = data['test']

	def run(self, name):
		print('')
		print('Regressor Fitting : ', name)
		self.regressor.fit(self.train['x'], self.train['y'])

	def validate(self):
		print('Regressor Validation')

		# R2 Score is implemented as a built-in feature
		# if R2 = 1.0, it means that model explains the data perfectly
		# if R2 < 0, the model hasnt trained properly
		r2 = self.regressor.score(self.test['x'], self.test['y'])

		# MAPE is Mean Absolute Percentage Error
		# Shows relative error% of forecasts
		ape = 0
		op = self.regressor.predict(self.test['x'])
		for i in range(len(op)):
			ape += abs((op[i] - float(self.test['y'].iloc[i]))*100.0/float(self.test['y'].iloc[i]))
		
		mape = ape/len(op)
		print('R2   : ', r2)
		print('MAPE : ', mape)
		print('')
		return {'r2': r2, 'mape': mape}
