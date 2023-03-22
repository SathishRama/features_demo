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
raw_cust_features_df = cust_df[['education','age','hh_income']]
computed_features_df = pd.DataFrame()
final_features_df = pd.DataFrame()
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

computed_features_df['num_bills'] = cust_df['cust_id'].apply(lambda x: feature_num_bills(x))
computed_features_df['num_paid_full'] = cust_df['cust_id'].apply(lambda x: feature_num_paid_full(x))

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

#during training
raw_cust_features_df = raw_cust_features_df.replace(cat_encoding)




final_features_df = pd.concat([raw_cust_features_df,computed_features_df],axis=1)
#print(final_features_df.head(10))
final_features_df.to_csv(data_dir+'/final/final_features.csv',index=False)

education = ["Associate's",	"Bachelor's",	"High School", "Masters"]

