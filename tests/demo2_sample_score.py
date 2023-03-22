import xgboost as xgb
import pickle
import pandas as pd
with open('../models/demo2_model.pkl', 'rb') as pickle_file:
    model = pickle.load(pickle_file)
input = pd.DataFrame({'education':3,'age_group':2,'income_group':4,'num_bills':2,'num_paid_full':1},index=[0])
print(model.get_booster().feature_names)
preds = model.predict_proba(input)[0]

print(preds)