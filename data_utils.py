import pandas as pd
from sklearn.preprocessing import StandardScaler
import chardet


def load_communities_and_crime_data():
  #load data
  with open('crimedata.csv','rb') as t:
    res = chardet.detect(t.read())
  data = pd.read_csv("crimedata.csv", na_values = ['?'], dtype = {'state':object, 'county':object, 'community':object, 'communityname':object}, encoding = res['encoding'])
  data.drop(['state', "countyCode", "communityCode","Êcommunityname", "nonViolPerPop"], axis = 1, inplace = True)
  for col in data.columns :
    data[col].fillna(0, inplace = True)
  return data


def load_electric_power_data():
  #load data
  with open('household_power_consumption.txt','rb') as t:
    res = chardet.detect(t.read())
  data = pd.read_csv("household_power_consumption.txt", sep=';', na_values = ['?'])
  data.drop(['Date', 'Time'], axis = 1, inplace = True)
  for col in data.columns :
    data[col].fillna(0, inplace = True)
  return data

def scale_data(data):
  scaler = StandardScaler()
  data_scaled = scaler.fit_transform(data)

  return data_scaled

def preprocess_data_crime(data):
  data_scaled = scale_data(data)
  X_scaled = data_scaled[:, :-1]
  y_scaled = data_scaled[:, -1]
  return X_scaled, y_scaled


def preprocess_data_electric(data):
  data_scaled = scale_data(data)
  X_scaled = data_scaled[:, :-1]  # Features: Voltage, Global_intensity, Sub_metering_1, Sub_metering_2, Sub_metering_3
  y_scaled = data_scaled[:, -1]  # Target: Global_active_power
#   print(f'data.shape : {data_scaled.shape}')
#   print(f'X_scaled.shape : {X_scaled.shape}')
#   print(f'y_scaled.shape : {y_scaled.shape}')
  return X_scaled, y_scaled