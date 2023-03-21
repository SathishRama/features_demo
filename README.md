# Unified Feature Engineering and Feature Serving Concept 
This is in Ideation phase and is work in progress. Hope to enhance it more once the concept is refined. 

This is a demo repo to illustrate how we can use the training feature pipelines in model serving code to compute features automatically without rewriting the code during model implementation. This is very basic demo and I intend to enhance it more as time permits. 

The thought here is, when we develop the model, we create feature pipelines,using a set of frameworks & coding standards, such that the computational logic can be re-used in serving code. We would need the Training and Serving environments to be same for this to be productive. In absence of that, we may need to create additional dataset loader framework that helps in abstracting the evnviroment nuances for the feature function logic ( more to come later on this )

In this demo example, I have a model that uses 3 features. 
  1.Customer ID
  2.Customer Age
  3.Customer Transactions Return Rate ( # of returns ) 

A basic xgboost classifier model is trained to predict probability of return for a new sale by a customer using previous customer sales & returns. 

I use a feature_mapping.json which defines the features metadata :
  1.Customer id feature is passed from the upstream invoking the model ( from api call )
  2.Customer Age is caculated using the Customer Id from the Customers Dataset
  3.Customer Return Rate is calcuated using a function from Transactions Dataset and customer id
  
  feature_mapping.json tells which features are to be taken from upstream and which features needs to be computed using logic from feature pipeline code.
  
  
