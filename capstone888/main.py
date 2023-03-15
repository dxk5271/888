import streamlit as st
import pandas as pd
import time
import scipy.io as sio
from scipy.io import loadmat
import pandas as pd
from sklearn.preprocessing import StandardScaler



st.title("Predict Cardiac Abnormalities")

sex = st.sidebar.radio(
    "What is your sex",
    ('Male', 'Female'))


ghealth = st.sidebar.slider('Rate your general health', 0, 10)
phealth = st.sidebar.slider('Rate your physical health', 0, 10)
mhealth = st.sidebar.slider('Rate your mental health', 0, 10)
activity = st.sidebar.slider('Rate how active you are', 0, 10)
st.sidebar.write('Have you ever had or have the below?')
bp_high = st.sidebar.radio("High Blood Pressure?", ('Yes', 'No'))
cholesterol_high = st.sidebar.radio("High Cholesterol?", ('Yes', 'No'))
stroke = st.sidebar.radio("Stroke?", ('Yes', 'No'))
asthma = st.sidebar.radio("Asthma?", ('Yes', 'No'))
cancer_skin = st.sidebar.radio("Skin Cancer?", ('Yes', 'No'))
cancer = st.sidebar.radio("Any other type of cancer", ('Yes', 'No'))
kidney_disease = st.sidebar.radio("Kidney Disease", ('Yes', 'No'))

diabetes = st.sidebar.radio("Diabetes", ('Yes', 'No'))
arthritis = st.sidebar.radio("Arthritis", ('Yes', 'No'))
difficulty_walking = st.sidebar.radio("Difficulty Walking?", ('Yes', 'No'))
smoker = st.sidebar.radio("Smoker", ('Yes', 'No'))

age = st.sidebar.slider('What is your age?', 18, 110)
bmi = st.sidebar.slider('BMI?', 5, 50)








x = """
sex                    438693 non-null  int32  
 1   general_health         438689 non-null  float64
 2   physical_health        438690 non-null  float64
 3   mental_health          438691 non-null  float64
 4   physical_activity      438693 non-null  int32  
 5   bp_high                438691 non-null  float64
 6   cholesterol_high       377857 non-null  float64
 7   stroke                 438691 non-null  float64
 8   asthma                 438693 non-null  int32  
 9   cancer_skin            438691 non-null  float64
 10  cancer_other           438690 non-null  float64
 11  kidney_disease         438690 non-null  float64
 12  diabetes               438690 non-null  float64
 13  arthritis              438690 non-null  float64
 14  difficulty_walking     420684 non-null  float64
 15  smoker                 417461 non-null  float64
 16  race **                  438693 non-null  int32  
 17  age                    438693 non-null  int32  
 18  bmi                    391841 non-null  float64
 19  education              438693 non-null  int32  
 20  income                 429846 non-null  float64
 21  alcohol_intake         407294 non-null  float64
 22  alcoholic              438693 non-null  int32  
 23  fruit_intake           404518 non-null  float64
 24  vegetable_intake       401698 non-null  float64
 25  french_fry_intake      400582 non-null  float64
 26  health_insurance       438693 non-null  int32  
 27  marital_status         438688 non-null  float64
 28  household_child_count  432558 non-null  float64
 29  vaccine_flu            411045 non-null  float64
 30  vaccine_pneumonia      409606 non-null  float64
 31  urban_status           431639 non-null  float64
 32  state                  438693 non-null  int32  
 33  target             

 """