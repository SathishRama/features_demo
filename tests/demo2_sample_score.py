import xgboost as xgb
import pickle
import pandas as pd
with open('../models/demo2_model.pkl', 'rb') as pickle_file:
    model = pickle.load(pickle_file)
input = pd.DataFrame({'age':50,'hh_income':40000},index=[0])
print(model.get_booster().feature_names)
preds = model.predict_proba(input)

print(preds)