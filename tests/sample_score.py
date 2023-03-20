import xgboost as xgb
import pickle
import pandas as pd
with open('../models/model.pkl', 'rb') as pickle_file:
    model = pickle.load(pickle_file)
input = pd.DataFrame({'cust_id':7,'cust_age':50,'r_rate':0.0},index=[0])
print(model.get_booster().feature_names)
preds = model.predict_proba(input)

print(preds)