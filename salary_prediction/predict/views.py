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
            'job_description_length': 2500,
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
                salary_predictor_path = os.path.join(BASE_DIR, 'linear_regression_pipeline.pkl')
                

                with open(salary_predictor_path, "rb") as file:
                    pipeline_pkl = joblib.load(file)



                input_df = pd.concat([cat_df, num_df], axis=1)

                pred = pipeline_pkl.predict(input_df)

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

            case "decision tree":
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                tree_model_path = os.path.join(BASE_DIR, 'amit_decision_tree_model.pkl')
                tree_encoder_path = os.path.join(BASE_DIR, 'amit_encoder.pkl')
                tree_model = joblib.load(tree_model_path)
                encoders = joblib.load(tree_encoder_path)

                for col in cat_df.columns:
                    cat_df[col] = encoders[col].transform(cat_df[col])

                input_df = pd.concat([cat_df, num_df], axis=1)

                prediction = round(tree_model.predict(input_df)[0], 2)
                print(prediction)
            case "random forest":
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                rf_pipeline_path = os.path.join(BASE_DIR, 'rf_regressor_pipeline.pkl')
                rf_pipeline = joblib.load(rf_pipeline_path)

                # Set default/fixed values
                fixed_features = {
                    "job_description_length": 2500,            # same as other models
                    "days_open": 7,                            # arbitrary default
                    "is_remote": 1 if numerical_features["remote_ratio"] >= 50 else 0,
                    "salary_currency": "USD"                   # model trained on USD
                }

                # Merge all features
                full_input = {
                    **numerical_features,
                    **categorical_features,
                    **fixed_features
                }

                # Final column order must match training columns
                input_df = pd.DataFrame([full_input])

                # Predict
                prediction = round(rf_pipeline.predict(input_df)[0], 2)

            
        print("Prediction:", prediction)

        
        return render(request, "form.html", {"predicted_salary": prediction})
    except Exception as e:
        print("Error during prediction:", e)
        return render(request, "form.html", {"error": f"Prediction failed: {e}"})

