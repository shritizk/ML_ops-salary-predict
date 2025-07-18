from django.shortcuts import render
import joblib
import pandas as pd

def predict_salar_with_linear_regression(request):
    
    prediction = None # final prediction just incase if something went wrong this will be send 

    # here we will take input of all the params
    input_dict = {

    }

    input_df = pd.DataFrame(input_dict)

    # load model 
    model = joblib.load('salary_predictor.pkl')
    encoder = joblib.load('encoder.pkl')

    # apply one hot to the new df
    input_enc = encoder.transform(input_df)

    pred = model.pridict(input_enc)
    
    print( pred) # for testing only

    prediction = round(pred[0],2)

    return render(request, "prediction.html",{prediction})


