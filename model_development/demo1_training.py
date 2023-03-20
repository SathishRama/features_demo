import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle,os, numpy as np
from xgboost import XGBClassifier

os.environ['my_env'] = 'Training'
#import features functions
from feature_pipelines.demo1_features import cust_df,return_rate_df,compute_r_rate_df

#create training train_data
compute_r_rate_df()
#print(return_rate_df)
train_df = pd.merge(cust_df[['cust_id','cust_age']],return_rate_df,on=['cust_id']).reset_index(drop=True)
labels_data = [1,1,1,0,0,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0]
y = pd.DataFrame(labels_data,columns=['label']).reset_index(drop=True)

# Split the train_data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=0.2, random_state=1)
print('Training Set:')
print(pd.concat([X_train,y_train],axis=1))
print('Test Set:')
print(pd.concat([X_test,y_test],axis=1))

clf=XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=20, objective='binary:logistic')
XGB=clf.fit(X_train,y_train)

#predict labels for test set
y_pred = clf.predict(X_test)
print("y_pred vs y_test:")
print(y_pred)
print(np.squeeze(y_test.values.tolist()))


# Print the accuracy score of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))

# save the model
pickle.dump(clf, open('../models/model.pkl', 'wb'))