import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor

def main():
	train_df = getData('data/train.csv')
	test_df = getData('data/test.csv')

	train_df = cleanTrainingData(train_df)
	train_df = cleanData(train_df)
	test_df = cleanData(test_df)

	model = train(train_df)

	predictions = makePredictions(model, test_df)

	createCsv(predictions)

def getData(path):
	data = pd.read_csv(path)

	if 'SalePrice' in data.columns:
		df = data[['Id', 'SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']]
	else:
		df = data[['Id', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']]
	
	return df

def cleanTrainingData(df):
	df[['SalePrice']] = np.log1p(df[['SalePrice']])
	df = df.loc[df['GrLivArea'] < 4500]

	return df

def cleanData(df):
	df['GrLivArea'] = np.log1p(df['GrLivArea'])

	df['HasBsmt'] = pd.Series(len(df['TotalBsmtSF']), index=df.index)
	df['HasBsmt'] = 0 
	df.loc[df['TotalBsmtSF']>0,'HasBsmt'] = 1
	df.loc[df['HasBsmt']==1,'TotalBsmtSF'] = np.log1p(df['TotalBsmtSF'])

	return df

def train(df):
	model = XGBRegressor()

	trainInput = df.drop(['SalePrice'], axis=1)
	trainOutput = df[['SalePrice']]

	model.fit(trainInput, trainOutput, verbose=False)
	predictions = model.predict(trainInput)

	print("Mean Absolute Training Error : " + str(mean_absolute_error(predictions, trainOutput)))

	return model

def makePredictions(model, input):
	predictions = model.predict(input)
	predictions = np.expm1(predictions)
	return predictions

def createCsv(predictions):
	rawTestData = pd.read_csv('data/test.csv')
	result = rawTestData
	result['SalePrice'] = predictions
	result[['Id', 'SalePrice']].to_csv("predictions.csv", index=False)

main()
