import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier  # Add sklearn import
import pipreqs  # Add pipreqs import

# Generate requirements.txt file
pipreqs.pipreqs("./", force=True)

# Load the trained model
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Credit Fraud Detection')

# Create input fields for key features
st.sidebar.header('Client Information')
gender = st.sidebar.radio('Gender', ['M', 'F'])
own_car = st.sidebar.radio('Owns Car', ['Y', 'N'])
own_property = st.sidebar.radio('Owns Property', ['Y', 'N'])
children = st.sidebar.number_input('Number of Children', min_value=0)
income = st.sidebar.number_input('Annual Income', min_value=0.0)
age_years = st.sidebar.slider('Age (years)', 20, 100)
employment_years = st.sidebar.slider('Employment Years', 0, 50)

# Convert age and employment duration to days format used in model
age_days = -age_years * 365
employment_days = -employment_years * 365

# Create feature dictionary
input_data = {
    'gender': gender,
    'own_car': own_car,
    'own_property': own_property,
    'children': children,
    'income': income,
    'age_in_days': age_days,
    'employment_in_days': employment_days,
    'work_phone': 0,  # Default value based on dataset analysis
    'phone': 0,       # Default value based on dataset analysis
    'email': 0,       # Default value based on dataset analysis
    'family_members': 1,  # Default value
    'months_balance': 0   # Default value
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Make prediction
if st.button('Predict Fraud Risk'):
    try:
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)[0][1]
        
        st.subheader('Prediction Results')
        st.write(f'Predicted class: {"High Risk" if prediction[0] == 1 else "Low Risk"}')
        st.write(f'Fraud Probability: {proba:.1%}')
        
        # Add explanation
        st.markdown("""
        **Interpretation:**
        - High Risk (1): Likely to default on credit payments
        - Low Risk (0): Likely to fulfill credit obligations
        """)
        
    except Exception as e:
        st.error(f'Error making prediction: {str(e)}')
