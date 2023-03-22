import os

import pandas as pd


def print_name(name,age):
    print(name, age)
    return 200

#func_name = "print_name"
#resp = globals()[func_name](*["Rama",40])
#print(resp)

os.environ['my_var'] = 'Training'
print(os.environ['my_var'])

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

print(feature_cat_encoding('education','Masters'))
