import pandas as pd
import os
#get the environment : Training vs Serving
try:
    env = os.environ['my_env']
    if env == 'Training':
        data_dir = '../data/train_data'
    if env == 'Serving':
        data_dir = '../data/serving_data'
except:
    data_dir = '../data/train_data'


#load datasets for the environment
cust_df =  pd.read_csv(data_dir+'/customers.csv')
tran_df =  pd.read_csv(data_dir+'/transactions.csv')

#Function to compute feature : return rate of customer
#Used both in Training Pipeline and Servig pipeline
def feature_func_r_rate(cust_id):
    #print("From feature_func_r_rate:",cust_id)
    total_trans = tran_df[tran_df['cust_id'] == cust_id]['tran_id'].count()
    num_returns = tran_df[tran_df['cust_id'] == cust_id][tran_df['tran_type'] == 'return']['tran_id'].count()
    #print(total_trans,num_returns)
    r_rate = num_returns/total_trans
    return r_rate

#Function to compute feature : age of customer
#Used both in Training Pipeline and Servig pipeline
def feature_func_age(cust_id):
    print("From feature_func_age:", cust_id)
    return cust_df[cust_df['cust_id'] == cust_id]['cust_age'].values[0]

#Training pipeline code
return_rate_df =  pd.DataFrame()
return_rate_df['cust_id'] = cust_df['cust_id']
return_rate_df['r_rate'] = 0.0

def compute_r_rate_df():
    return_rate_df['r_rate'] = return_rate_df['cust_id'].apply(lambda x: feature_func_r_rate(x))
    #print(return_rate_df.head(10))
    return

if __name__ == "__main__":
    compute_r_rate_df()


