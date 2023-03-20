from flask import Flask,request,jsonify
import pandas as pd
import json,pickle
import os
from feature_pipelines.demo1_features import  feature_func_r_rate, feature_func_age

app = Flask(__name__)
os.environ['my_env'] = 'Serving'

def get_model_metadata(model):
    model_metadata_file = open('../metadata/feature_mapping.json')
    model_metadata = json.load(model_metadata_file)
    print(model_metadata['demo'])
    return model_metadata['demo']


def get_features(features,api_data):
    feature_list = {}
    for feature_name in features:
        feature_source = features[feature_name]['feature_source']
        if feature_source == 'api':
            feature_list[feature_name] = int(api_data[features[feature_name]['feature_api_field']])

        elif feature_source == 'logic':
            #call the function to calculate feature
            try:
                feature_func = features[feature_name]['feature_logic_func']
                func_inputs = features[feature_name]['feature_func_inputs']
                func_inputs_values = []
                for input in func_inputs:
                    func_inputs_values.append(feature_list[input])
                print("Calling func {feature_func} with values :",*func_inputs_values)

                feature_list[feature_name] = globals()[feature_func](*func_inputs_values)
                print('Feature Details:', feature_list )

            except Exception as e:
                print(e)
        else:
            print("Error : Unknown feature source type")
    return feature_list

@app.route('/', methods=['POST'])
def read_request():
    try:
        api_data = request.get_json()
        #cust_id = train_data['cust_id']
        model_metadata = get_model_metadata('demo')

        # load model using pickle
        model_name = model_metadata['model_file']
        with open('../models/'+model_name, 'rb') as pickle_file:
            model = pickle.load(pickle_file)

        # get features ( Demo Features : cust_id, cust_age and return rate )
        features = get_features(model_metadata['features'],api_data)
        print("Features  :", features)
        feature_values = pd.DataFrame(features,index=[0])
        print("Features Retrieved :",feature_values )

        # make prediction
        prediction = model.predict_proba(feature_values)[0]
        print("Prediction:", prediction)
        prediction = str(prediction)
        print("Prediction:",prediction)

        #give response
        return ({'Status' : 'Success',
                  'Prediction Probs': prediction})
    except Exception as e:
        print("Exception occurred:",e)
        return jsonify({'message': 'Bad Request'})

if __name__ == '__main__':
    app.run(debug=True,port=5000)