import pandas as pd
import os
os.environ['my_env'] = 'Training'
## Pre compute 1 feature : number of bills paid in full ( through a batch process )

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
payments_df = pd.read_csv(data_dir + '/card_payments.csv')

payments_summary_df = pd.DataFrame()
payments_summary_df = payments_df
payments_summary_df.loc[(payments_summary_df['balance'] <= payments_summary_df['pay_amount']) , 'paid_full'] = 1
payments_summary_df.loc[(payments_summary_df['balance'] > payments_summary_df['pay_amount']) , 'paid_full'] = 0
payments_summary_df = payments_summary_df[['card_id','bill_date','paid_full']]
payments_summary_df.to_csv(data_dir+'/pre_computed/'+ 'payments_summary.csv',index=False)