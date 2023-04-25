import streamlit as st
import pandas as pd
import time
import scipy.io as sio
from scipy.io import loadmat
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgbm
import time

from sklearn import metrics
import time
from scipy.stats import ks_2samp
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay
import streamlit as st
from PIL import Image

image = Image.open('heart.jpeg')

st.sidebar.image(image)
print('hi')

st.title("Are you at risk for heart disease?")
st.markdown("""
        Directions: to predict your heart disease status, please follow the steps below:
        1. Enter the parameters that best describe you on the left;
        2. Select a predictive model as your "Doctor";
        2. Press the "Submit" button and wait for the result.
            
        **Keep in mind that this result is not equivalent to a medical diagnosis!**

        If the result indicates that you are at risk of heart disease,
        we recommend you receive follow-up chest screening and/or other medical tests to further confirm the heart disease's existence and severity.
         
        """)

st.markdown("""
        ***Author: Daniel King, Lin Tang, Kannan Sundaram, Tyler Staffin, Mark Stier (DAAN888-SP23-Group3)*** 
        """)


st.sidebar.write('Have you ever had or have the below?')


#state = st.sidebar.radio("State", ('X', 'Y'))


sex = st.sidebar.radio("What is your sex", ('Male', 'Female'))

def sex_m(sex):
    if sex == 'Female':
        sex_m = 0
    else:
        sex_m = 1
    return sex_m

sex_v = sex_m(sex)

age = st.sidebar.slider('What is your age?', 18, 120)

if age >= 80:
    age_2=1
    age_3=1
    age_4=1
    age_5=1
    age_6 =1
    age_7 =1
    age_8 =1
    age_9=1
    age_10 =1
    age_11 =1
    age_12 =1
    age_13 = 1
elif age < 80 and age >= 74:
  
    age_2=1
    age_3=1
    age_4=1
    age_5=1
    age_6 =1
    age_7 =1
    age_8 =1
    age_9=1
    age_10 =1
    age_11 =1
    age_12 =1
    age_13 = 0
elif age < 74 and age >= 70:
  
    age_2=1
    age_3=1
    age_4=1
    age_5=1
    age_6 =1
    age_7 =1
    age_8 =1
    age_9=1
    age_10 =1
    age_11 =1
    age_12 =0
    age_13 = 0
elif age < 69 and age >= 65:
  
    age_2=1
    age_3=1
    age_4=1
    age_5=1
    age_6 =1
    age_7 =1
    age_8 =1
    age_9=1
    age_10 =1
    age_11 =0
    age_12 = 0
    age_13 = 0

elif age < 65 and age >= 60:
  
    age_2=1
    age_3=1
    age_4=1
    age_5=1
    age_6 =1
    age_7 =1
    age_8 =1
    age_9=1
    age_10 =0
    age_11 =0
    age_12 = 0
    age_13 = 0
elif age < 60 and age >= 55:
  
    age_2=1
    age_3=1
    age_4=1
    age_5=1
    age_6 =1
    age_7 =1
    age_8 =1
    age_9=0
    age_10 =0
    age_11 =0
    age_12 = 0
    age_13 = 0
elif age < 55 and age >= 50:
  
    age_2=1
    age_3=1
    age_4=1
    age_5=1
    age_6 =1
    age_7 =1
    age_8 =0
    age_9=0
    age_10 =0
    age_11 =0
    age_12 = 0
    age_13 = 0
elif age < 50 and age >= 45:
  
    age_2=1
    age_3=1
    age_4=1
    age_5=1
    age_6 =1
    age_7 =0
    age_8 =0
    age_9=0
    age_10 =0
    age_11 =0
    age_12 = 0
    age_13 = 0

elif age < 45 and age >= 40:
  
    age_2=1
    age_3=1
    age_4=1
    age_5=1
    age_6 =0
    age_7 =0
    age_8 =0
    age_9=0
    age_10 =0
    age_11 =0
    age_12 = 0
    age_13 = 0

elif age < 40 and age >= 35:
  
    age_2=1
    age_3=1
    age_4=1
    age_5=0
    age_6 =0
    age_7 =0
    age_8 =0
    age_9=0
    age_10 =0
    age_11 =0
    age_12 = 0
    age_13 = 0

elif age < 35 and age >= 30:
  
    age_2=1
    age_3=1
    age_4=0
    age_5=0
    age_6 =0
    age_7 =0
    age_8 =0
    age_9=0
    age_10 =0
    age_11 =0
    age_12 = 0
    age_13 = 0

elif age < 30 and age >= 25:
  
    age_2=1
    age_3=0
    age_4=0
    age_5=0
    age_6 =0
    age_7 =0
    age_8 =0
    age_9=0
    age_10 =0
    age_11 =0
    age_12 = 0
    age_13 = 0

else:
    age_2=0
    age_3=0
    age_4=0
    age_5=0
    age_6 =0
    age_7 =0
    age_8 =0
    age_9=0
    age_10 =0
    age_11 =0
    age_12 = 0
    age_13 = 0


race = st.sidebar.radio("Race", ('White', 'Black','Asian','American Indian/Alaskan Native','Hispanic', 'Other'))

if race == 'White':
    race_1 = 1
    race_2 = 0
    race_3 = 0
    race_4 = 0
    race_5 = 0
    race_6 = 0 
elif race == 'Black':
    race_1 = 0
    race_2 = 1
    race_3 = 0
    race_4 = 0
    race_5 = 0
    race_6 = 0 
elif race == 'Asian':
    race_1 = 0
    race_2 = 0
    race_3 = 1
    race_4 = 0
    race_5 = 0
    race_6 = 0 
elif race == 'American Indian/Alaskan Native':
    race_1 = 0
    race_2 = 0
    race_3 = 0
    race_4 = 1
    race_5 = 0
    race_6 = 0
elif race == 'Hispanic':
    race_1 = 0
    race_2 = 0
    race_3 = 0
    race_4 = 0
    race_5 = 1
    race_6 = 0 
elif race == 'Other':
    race_1 = 0
    race_2 = 0
    race_3 = 0
    race_4 = 0
    race_5 = 0
    race_6 = 1 

education = st.sidebar.radio("Education Level", ('Did not graduate High School','Graduated High School', 'Attended College or Technical School', 'Graduated College or Technical School'))

if education == 'Did not graduate High School':
    education_2 = 0
    education_3 = 0
    education_4 = 0

elif education == 'Graduated High School':
    education_2 = 1
    education_3 = 0
    education_4 = 0

elif education == 'Attended College or Technical School':
    education_2 = 1
    education_3 = 1
    education_4 = 0
elif education == 'Graduated College or Technical School':
    education_2 = 1
    education_3 = 1
    education_4 = 1

marital_status = st.sidebar.radio("Martial Status", ('Married', 'Divorce','Widowed','Seperated','Never Married','Unmarried Couple'))

if marital_status == 'Married':
     marital_status_1 = 1 
     marital_status_2 = 0
     marital_status_3 = 0
     marital_status_4 = 0
     marital_status_5 = 0
     marital_status_6 = 0
elif marital_status == 'Divorce':
     marital_status_1 = 0
     marital_status_2 = 1
     marital_status_3 = 0
     marital_status_4 = 0
     marital_status_5 = 0
     marital_status_6 = 0
elif marital_status == 'Widowed':
     marital_status_1 = 0
     marital_status_2 = 0
     marital_status_3 = 1
     marital_status_4 = 0
     marital_status_5 = 0
     marital_status_6 = 0
elif marital_status == 'Seperated':
     marital_status_1 = 0
     marital_status_2 = 0
     marital_status_3 = 0
     marital_status_4 = 1
     marital_status_5 = 0
     marital_status_6 = 0
elif marital_status == 'Never Married':
     marital_status_1 = 0
     marital_status_2 = 0
     marital_status_3 = 0
     marital_status_4 = 0
     marital_status_5 = 1
     marital_status_6 = 0
elif marital_status == 'Unmarried Couple':
     marital_status_1 = 0
     marital_status_2 = 0
     marital_status_3 = 0
     marital_status_4 = 0
     marital_status_5 = 0
     marital_status_6 = 1

income = st.sidebar.slider('Annual Household Income(in thousands)', 0, 250)
if income < 10:
    income_2 = 0
    income_3 = 0
    income_4 = 0
    income_5 = 0 
    income_6 = 0
    income_7 = 0
    income_8 = 0
    income_9 = 0
    income_10 = 0
    income_11 = 0

elif income < 15 and income >= 10:
    income_2 = 1
    income_3 = 0
    income_4 = 0
    income_5 = 0 
    income_6 = 0
    income_7 = 0
    income_8 = 0
    income_9 = 0
    income_10 = 0
    income_11 = 0

elif income < 20 and income >= 15:
    income_2 = 1
    income_3 = 1
    income_4 = 0
    income_5 = 0 
    income_6 = 0
    income_7 = 0
    income_8 = 0
    income_9 = 0
    income_10 = 0
    income_11 = 0

elif income < 25 and income >= 20:
    income_2 = 1
    income_3 = 1
    income_4 = 1
    income_5 = 0 
    income_6 = 0
    income_7 = 0
    income_8 = 0
    income_9 = 0
    income_10 = 0
    income_11 = 0

elif income < 35 and income >= 25:
    income_2 = 1
    income_3 = 1
    income_4 = 1
    income_5 = 1 
    income_6 = 0
    income_7 = 0
    income_8 = 0
    income_9 = 0
    income_10 = 0
    income_11 = 0

elif income < 50 and income >= 35:
    income_2 = 1
    income_3 = 1
    income_4 = 1
    income_5 = 1 
    income_6 = 1
    income_7 = 0
    income_8 = 0
    income_9 = 0
    income_10 = 0
    income_11 = 0

elif income < 75 and income >= 50:
    income_2 = 1
    income_3 = 1
    income_4 = 1
    income_5 = 1 
    income_6 = 1
    income_7 = 1
    income_8 = 0
    income_9 = 0
    income_10 = 0
    income_11 = 0

elif income < 100 and income >= 75:
    income_2 = 1
    income_3 = 1
    income_4 = 1
    income_5 = 1 
    income_6 = 1
    income_7 = 1
    income_8 = 1
    income_9 = 0
    income_10 = 0
    income_11 = 0

elif income < 150 and income >= 100:
    income_2 = 1
    income_3 = 1
    income_4 = 1
    income_5 = 1 
    income_6 = 1
    income_7 = 1
    income_8 = 1
    income_9 = 1
    income_10 = 0
    income_11 = 0

elif income < 200 and income >= 150:
    income_2 = 1
    income_3 = 1
    income_4 = 1
    income_5 = 1 
    income_6 = 1
    income_7 = 1
    income_8 = 1
    income_9 = 1
    income_10 = 1
    income_11 = 0

else:
    income_2 = 1
    income_3 = 1
    income_4 = 1
    income_5 = 1 
    income_6 = 1
    income_7 = 1
    income_8 = 1
    income_9 = 1
    income_10 = 1
    income_11 = 1

bmi = st.sidebar.slider('BMI?', 10, 99)
bmi_v = bmi

urban_status = st.sidebar.radio("Living", ('Urban', 'Rural'))

def urban_status_m(urban_status):
    if urban_status == 'Urban':
        urban_status_m = 1
    else:
        urban_status_m = 2
    return urban_status_m

urban_status_v = urban_status_m(urban_status)


household_child_count = st.sidebar.slider('Children in Household', 0, 20)
household_child_count_v = household_child_count

ghealth = st.sidebar.slider('Rate your general health(1=Poor; 5=Excellect)', 1, 5)

if ghealth == 1:
    general_health_fair = 0
    general_health_good = 0
    general_health_verygood = 0
    general_health_excellent = 0
elif ghealth == 2:
    general_health_fair = 1
    general_health_good = 0
    general_health_verygood = 0
    general_health_excellent = 0
elif ghealth == 3:
    general_health_fair = 1
    general_health_good = 1
    general_health_verygood = 0
    general_health_excellent = 0
elif ghealth == 4:
    general_health_fair = 1
    general_health_good = 1
    general_health_verygood = 1
    general_health_excellent = 0
elif ghealth == 5:
    general_health_fair = 1
    general_health_good = 1
    general_health_verygood = 1
    general_health_excellent = 1  


def yesno(var):
    if var == 'Yes':
        yesno = 1
    else:
        yesno = 0
    return yesno

physical_activity = st.sidebar.radio("Did you have physical activity or exercise during the past 30 days other than their regular job?", ('No','Yes'))
physical_activity_v = yesno(physical_activity)

cholesterol_high = st.sidebar.radio("High Cholesterol?", ('No', 'Yes')) #1 is yes #0 no
cholesterol_high_v = yesno(cholesterol_high)

stroke = st.sidebar.radio("Stroke?", ('No', 'Yes'))
stroke_v = yesno(stroke)


asthma = st.sidebar.radio("Asthma?", ('No', 'Yes'))
asthma_v = yesno(asthma)

cancer_skin = st.sidebar.radio("Skin Cancer?", ('No', 'Yes'))
cancer_skin_v = yesno(cancer_skin)

cancer = st.sidebar.radio("Any other type of cancer", ('No', 'Yes'))
cancer_other_v = yesno(cancer)

kidney_disease = st.sidebar.radio("Kidney Disease", ('No', 'Yes'))
kidney_disease_v = yesno(kidney_disease)

arthritis = st.sidebar.radio("Arthritis", ('No', 'Yes'))
arthritis_v = yesno(arthritis)

difficulty_walking = st.sidebar.radio("Difficulty Walking?", ('No', 'Yes'))
difficulty_walking_v = yesno(difficulty_walking)

smoker = st.sidebar.radio("Have you ever smoked at least 100 cigarettes?", ('No', 'Yes'))
smoker_v = yesno(smoker)

alcoholic = st.sidebar.radio("Heavy drinkers(adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)", ('No', 'Yes'))
alcoholic_v = yesno(alcoholic)

health_insurance = st.sidebar.radio("Do you have some form of Health Insurance?", ('No', 'Yes'))
health_insurance_v = yesno(health_insurance)


vaccine_flu = st.sidebar.radio("Did you receive a Flu shot/spray in the past 12 months?", ('No', 'Yes'))
vaccine_flu_v = yesno(vaccine_flu)

vaccine_pneumonia = st.sidebar.radio("Did you receive pneumonia shot ever?", ('No', 'Yes'))
vaccine_pneumonia_v = yesno(vaccine_pneumonia)


bp_high = st.sidebar.radio("High Blood Pressure?", ('Yes', 'Yes, but only when pregnant', 'No','Borderline High'))

if bp_high == "Yes":
    bp_high_1 = 1
    bp_high_2 = 0
    bp_high_3 = 0
    bp_high_4 = 0
elif  bp_high == "Yes, but only when pregnant":
    bp_high_1 = 0
    bp_high_2 = 1
    bp_high_3 = 0
    bp_high_4 = 0
elif  bp_high == "No":
    bp_high_1 = 0
    bp_high_2 = 0
    bp_high_3 = 1
    bp_high_4 = 0
elif  bp_high == "Borderline High":
    bp_high_1 = 0
    bp_high_2 = 0
    bp_high_3 = 0
    bp_high_4 = 1

diabetes = st.sidebar.radio("Diabetes", ('Yes', 'Yes, but only during pregnancy','No','Prediabetic'))

if diabetes == 'Yes':

   diabetes_1 = 1
   diabetes_2 = 0
   diabetes_3 = 0
   diabetes_4 = 0
elif diabetes == 'Yes, but only during pregnancy':
   diabetes_1 = 0
   diabetes_2 = 1
   diabetes_3 = 0
   diabetes_4 = 0
elif diabetes == 'No':
   diabetes_1 = 0
   diabetes_2 = 0
   diabetes_3 = 1
   diabetes_4 = 0
elif diabetes == 'Prediabetic':
   diabetes_1 = 0
   diabetes_2 = 0
   diabetes_3 = 0
   diabetes_4 = 1


phealth = st.sidebar.slider('Number of days you were physcially unwell in last 30 days ', 0, 30)
phealth_v = phealth
mhealth = st.sidebar.slider('Number of days you were mentally unwell in last 30 days ', 0, 30)
mhealth_v = mhealth

alcohol_intake = st.sidebar.slider('Avg alcoholic drinks per day in past 30 days:', 0, 50)
alcohol_intake_v = alcohol_intake
fruit_intake = st.sidebar.slider('On average, how many times do you eat fruit per day?', 0, 20)
fruit_intake_v = fruit_intake

vegetable_intake = st.sidebar.slider('On average, how many times do you eat dark vegetable per day?', 0, 20)
vegetable_intake_v = vegetable_intake
french_fry_intake = st.sidebar.slider('On average, how many times do you eat French Fry per day?', 0, 20)
french_fry_intake_v = french_fry_intake


selected_models = st.selectbox('Select your Predictive Doctor:', ['Dr.LightGBM_calibrated','Dr.LightGBM','Dr.Logistic Regression'])


with st.spinner('Calculating...'):
    if st.button('Submit'):
        modlist = []
        modlist.append(sex_v)
        modlist.append(physical_activity_v)
        modlist.append(cholesterol_high_v)
        modlist.append(stroke_v)
        modlist.append(asthma_v)
        modlist.append(cancer_skin_v)
        modlist.append(cancer_other_v)
        modlist.append(kidney_disease_v)
        modlist.append(arthritis_v)
        modlist.append(difficulty_walking_v)
        modlist.append(smoker_v)
        modlist.append(alcoholic_v)
        modlist.append(health_insurance_v)
        modlist.append(vaccine_flu_v)
        modlist.append(vaccine_pneumonia_v)
        modlist.append(urban_status_v)
        modlist.append(general_health_fair)
        modlist.append(general_health_good)
        modlist.append(general_health_verygood)
        modlist.append(general_health_excellent)
        modlist.append(education_2)
        modlist.append(education_3)
        modlist.append(education_4)
        modlist.append(age_2)
        modlist.append(age_3)
        modlist.append(age_4)
        modlist.append(age_5)
        modlist.append(age_6)
        modlist.append(age_7)
        modlist.append(age_8)
        modlist.append(age_9)
        modlist.append(age_10)
        modlist.append(age_11)
        modlist.append(age_12)
        modlist.append(age_13)
        modlist.append(income_2)
        modlist.append(income_3)
        modlist.append(income_4)
        modlist.append(income_5)
        modlist.append(income_6)
        modlist.append(income_7)
        modlist.append(income_8)
        modlist.append(income_9)
        modlist.append(income_10)
        modlist.append(income_11)
        modlist.append(bp_high_1)
        modlist.append(bp_high_2)
        modlist.append(bp_high_3)
        modlist.append(bp_high_4)
        modlist.append(diabetes_1)
        modlist.append(diabetes_2)
        modlist.append(diabetes_3)
        modlist.append(diabetes_4)
        modlist.append(race_1)
        modlist.append(race_2)
        modlist.append(race_3)
        modlist.append(race_4)
        modlist.append(race_5)
        modlist.append(race_6)
        modlist.append(marital_status_1)
        modlist.append(marital_status_2)
        modlist.append(marital_status_3)
        modlist.append(marital_status_4)
        modlist.append(marital_status_5)
        modlist.append(marital_status_6)
        modlist.append(phealth_v)
        modlist.append(mhealth_v)
        modlist.append(bmi_v)
        modlist.append(fruit_intake_v)
        modlist.append(vegetable_intake_v)
        modlist.append(fruit_intake_v)
        modlist.append(household_child_count_v)
        modlist.append(alcohol_intake_v)

        df = pd.DataFrame(modlist).T
        df.columns = [f"column{i+1}" for i in range(len(df.columns))]

        # read training data
        X_train = pd.read_parquet('ori_test_0218.parquet',)
        continuous_var = ['physical_health', 'mental_health', 'bmi', 'fruit_intake', 'vegetable_intake', 'french_fry_intake',
                  'household_child_count', 'alcohol_intake']

        # drop target
        X_train = X_train.drop('target',axis=1)

        df.columns = list(X_train.columns)

        # scaler input data
        scaler = MinMaxScaler()
        # train scaler on train data and re-scale test data
        X_train[continuous_var] = scaler.fit_transform(X_train[continuous_var])

        df[continuous_var] = scaler.transform(df[continuous_var])

        # Model 1 - LR
        if selected_models == 'Dr.Logistic Regression':
          loaded_model = pickle.load(open('lr1.pkl', 'rb'))
          result = loaded_model.predict(df)
          proba = loaded_model.predict_proba(df)[0][1]

          if result == 0:
               st.write("**This Patient is NOT at Risk For Heart Disease**")
               st.markdown(f"**The probability that you have"
                        f" heart disease is {round(proba * 100, 2)}%."
                        f" You are healthy!**")  
               st.image("healthy_heart.jpg",
               caption="Your heart seems to be okay! - Dr. Logistic Regression")

          if result == 1:
              st.write("**This Patient is at Risk For Heart Disease**")
              st.markdown(f"**The probability that you have"
                        f" heart disease is {round(proba * 100, 2)}%."
                        f" It sounds like you are not healthy.**")
              st.image("unhealthy_heart.jpg",
              caption="I'm worried about your heart, it needs more care! - Dr. Logistic Regression")


        # Model 2 - LightGBM
        if selected_models == 'Dr.LightGBM':
          loaded_model = pickle.load(open('lgbm4.pkl', 'rb'))
          result = loaded_model.predict(df)
          proba = loaded_model.predict_proba(df)[0][1]

          if result == 0:
               st.write("**This Patient is NOT at Risk For Heart Disease**")
               st.markdown(f"**The probability that you have"
                        f" heart disease is {round(proba * 100, 2)}%."
                        f" You are healthy!**") 
               st.image("healthy_heart.jpg",
               caption="Your heart seems to be okay! - Dr. Light GBM")
 
          if result == 1:
              st.write("**This Patient is at Risk For Heart Disease**")
              st.markdown(f"**The probability that you have"
                        f" heart disease is {round(proba * 100, 2)}%."
                        f" It sounds like you are not healthy.**")
              st.image("unhealthy_heart.jpg",
              caption="I'm worried about your heart, it needs more care! - Dr. Light GBM")



        if selected_models == 'Dr.LightGBM_calibrated':
          loaded_model = pickle.load(open('lgbm5.pkl', 'rb'))
          
        #result = loaded_model.predict(df)
          result = loaded_model.predict_proba(df)[0][1]
          real_pos_prob = 0.08137746727104128
          if result <= real_pos_prob:
               st.write("**This Patient is NOT at Risk For Heart Disease**")
               st.markdown(f"**The probability that you have"
                        f" heart disease is {round(result * 100, 2)}%."
                        f" You are healthy!**") 
               st.image("healthy_heart.jpg",
               caption="Your heart seems to be okay! - Dr. Light GBM_calibrated")
 
          if result > real_pos_prob:
              st.write("**This Patient is at Risk For Heart Disease**")
              st.markdown(f"**The probability that you have"
                        f" heart disease is {round(result * 100, 2)}%."
                        f" It sounds like you are not healthy.**")
              st.image("unhealthy_heart.jpg",
              caption="I'm worried about your heart, it needs more care! - Dr. LightGBM_calibrated")

