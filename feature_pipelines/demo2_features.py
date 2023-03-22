import pandas as pd
import os
os.environ['my_env'] = 'Training'
from _datetime import datetime
#get the environment : Training vs Serving
try:
    env = os.environ['my_env']
    if env == 'Training':
        data_dir = '../data/train_data/demo2'
    if env == 'Serving':
        data_dir = '../data/serving_data/demo2'
except:
    data_dir = '../data/train_data/demo2'

# load datasets for the environment
cust_df = pd.read_csv(data_dir + '/raw/customers_v2.csv')
cards_df = pd.read_csv(data_dir + '/raw/cards.csv')
payments_df = pd.read_csv(data_dir + '/raw/card_payments.csv')
payments_summary_df = pd.read_csv(data_dir + '/pre_computed/payments_summary.csv')

#global variables or attributes
current_date = datetime.date(datetime.now())

# print(cust_df.head())
# print(cards_df.head())
# print(payments_df.head())
# print(payments_summary_df.head())

#features
# cust_id
# education
# age
# customer_since  - computed
# hh_income
# num_bills  - computed
# num_paid_full - computed

# print(raw_cust_features_df.head())

def feature_num_bills(cust_id):
    card_id = cards_df[cards_df['cust_id'] == cust_id]['card_id'].values[0]
    #print(card_id)
    num_bills = payments_df[payments_df['card_id'] == card_id].count()['bill_date']
    return num_bills

def feature_num_paid_full(cust_id):
    card_id = cards_df[cards_df['cust_id'] == cust_id]['card_id'].values[0]
    num_paid_full = payments_summary_df[payments_summary_df['card_id'] == card_id].sum()['paid_full']
    return num_paid_full

#encode categorial feature : education ( High shool, Associate, Bachelors, Masters )
cat_encoding = {"education":
                    {"Associate's": 2,
                     "Bachelor's": 3,
                     "High School": 1,
                     "Masters": 4
                     }
                }
#created below function to help with serving
def feature_cat_encoding(feature_name,feature_value):
    return cat_encoding[feature_name][feature_value]

def feature_age_group(age):
    print('from func age group:',age)
    age = int(age)
    age_group = 0
    if age in range(16,24):
        age_group =  1
    elif age in range(24,34):
        age_group = 2
    elif age in range(34, 44):
        age_group = 3
    elif age in range(44, 100):
        age_group = 4
    return age_group

def feature_income_group(income):
    print('from func age group:',income)
    income = int(income)
    income_group = 0
    if income in range(0,10000):
        income_group =  1
    elif income in range(10000,20000):
        income_group = 2
    elif income in range(20000, 30000):
        income_group = 3
    elif income in range(30000, 50000):
        income_group = 4
    elif income in range(50000, 80000):
        income_group = 5
    elif income in range(80000, 500000):
        income_group = 6
    return income_group

def compute_features_training():
    raw_cust_features_df = cust_df[['education', 'age', 'hh_income']]
    computed_features_df = pd.DataFrame()
    final_features_df = pd.DataFrame()
    computed_features_df['num_bills'] = cust_df['cust_id'].apply(lambda x: feature_num_bills(x))
    computed_features_df['num_paid_full'] = cust_df['cust_id'].apply(lambda x: feature_num_paid_full(x))
    #during training
    raw_cust_features_df = raw_cust_features_df.replace(cat_encoding)
    raw_cust_features_df['age_group'] = raw_cust_features_df['age'].apply(lambda x: feature_age_group(x))
    raw_cust_features_df['income_group'] = raw_cust_features_df['hh_income'].apply(lambda x: feature_income_group(x))
    raw_cust_features_df = raw_cust_features_df.drop(columns=['age','hh_income'])

    final_features_df = pd.concat([raw_cust_features_df,computed_features_df],axis=1)
    #print(final_features_df.head(10))
    final_features_df.to_csv(data_dir+'/final/final_features.csv',index=False)

if __name__ == "__main__":
    compute_features_training()