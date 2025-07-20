from django.shortcuts import render , redirect
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages

import joblib
import pandas as pd
import os


@csrf_exempt
def form(request):
    return render(request,"form.html")

def home(request):
    return render(request,"home.html")

@csrf_exempt
def predict_salar_with_linear_regression(request):
    try:
        
        numerical_features = {
            'remote_ratio': float(request.POST.get('remote_ratio', 0)),
            'years_experience': float(request.POST.get('years_experience', 0)),
            'job_description_length': float(request.POST.get('job_description_length', 0)),
            'benefits_score': float(request.POST.get('benefits_score', 0)),
        }

        categorical_features = {
            'job_title': request.POST.get('job_title', ''),
            'experience_level': request.POST.get('experience_level', '').strip(),
            'employment_type': request.POST.get('employment_type', '').strip(),
            'company_location': request.POST.get('company_location', '').strip(),
            'company_size': request.POST.get('company_size', '').strip(),
            'employee_residence': request.POST.get('employee_residence', '').strip(),
            'education_required': request.POST.get('education_required', '').strip(),
            'industry': request.POST.get('industry', '').strip(),
        }

        
        cat_df = pd.DataFrame([categorical_features])
        num_df = pd.DataFrame([numerical_features])
        
        algo_selected = request.POST.get("algorithm")
        match algo_selected:
            case "linear":
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                salary_predictor_path = os.path.join(BASE_DIR, 'salary_predictor.pkl')
                encoder_path = os.path.join(BASE_DIR, 'encoder.pkl')
                with open(salary_predictor_path, "rb") as file:
                    salary_predict_pkl = joblib.load(file)
                with open(encoder_path, "rb") as file:
                    enc_pkl = joblib.load(file)
                # Transform categorical input with encoder
                encoded_cat = enc_pkl.transform(cat_df)
                # Convert sparse matrix to dense array
                encoded_cat_array = encoded_cat.toarray()
                # Get feature names after encoding
                feature_names = enc_pkl.get_feature_names_out(cat_df.columns)

                # Create DataFrame from encoded categorical features with proper columns
                encoded_cat_df = pd.DataFrame(encoded_cat_array, columns=feature_names)

                # Combine encoded categorical features with numerical features
                input_df = pd.concat([encoded_cat_df, num_df], axis=1)

                # Predict salary
                pred = salary_predict_pkl.predict(input_df)
                prediction = round(pred[0], 2)

            case "xgboost":
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                xgb_pipeline_path = os.path.join(BASE_DIR, 'xgboost_salary_model.pkl')

                # Load the saved pipeline (only once ideally, but can load here)
                xgb_pipeline = joblib.load(xgb_pipeline_path)

                #df 
                input_df = pd.concat([cat_df, num_df])

                # Then use this pipeline to predict on your processed input dataframe
                prediction = round(xgb_pipeline.predict(input_df)[0], 2)
                
            
        print("Prediction:", prediction)

        
        return render(request, "form.html", {"predicted_salary": prediction})
    except Exception as e:
        print("Error during prediction:", e)
        return render(request, "form.html", {"error": f"Prediction failed: {e}"})

