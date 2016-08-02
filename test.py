# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

#def get_data(file_name):
#	data = pd.read_csv(file_name)
#	X_Parameter = []
#	Y_Parameter = []
#	for single_square_feet, single_price_value in zip(data['square_feet'], data['price']):
#		X_Parameter.append(float(single_square_feet))
#		Y_Parameter.append(float(single_price_value))
#	return X_Parameter, Y_Parameter
	
X = [[150.0], [200.0], [250.0], [300.0], [350.0], [400.0], [600.0]]
Y = [6450.0, 7450.0, 8450.0, 9450.0, 11450.0, 15450.0, 18450.0]
print X
print Y

def linear_model_main(X_parameters,Y_parameters,predict_value):

	# Create linear regression object
	regr = linear_model.LinearRegression()
	regr.fit(X_parameters, Y_parameters)
	predict_outcome = regr.predict(predict_value)
	predictions = {}
	predictions['intercept'] = regr.intercept_
	predictions['coefficient'] = regr.coef_
	predictions['predicted_value'] = predict_outcome
	return predictions 

def show_linear_line(X_parameters,Y_parameters):
# Create linear regression object
	regr = linear_model.LinearRegression()
	regr.fit(X_parameters, Y_parameters)
	plt.scatter(X_parameters,Y_parameters,color='blue')
	plt.plot(X_parameters,regr.predict(X_parameters),color='red',linewidth=3)
	# plt.xticks(())
	# plt.yticks(())
	plt.show() 

predictvalue = 700
result = linear_model_main(X,Y,predictvalue)
print "Intercept value " , result['intercept']
print "coefficient" , result['coefficient']
print "Predicted value: ",result['predicted_value']



X.append([700])
Y.append(float(result['predicted_value']))
print X
print Y

show_linear_line(X, Y)