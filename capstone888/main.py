import streamlit as st
import pandas as pd
import time
import scipy.io as sio
from scipy.io import loadmat
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgbm

from sklearn import metrics
import time
from scipy.stats import ks_2samp
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay



st.title("Are you at risk for heart disease?")




st.sidebar.write('Have you ever had or have the below?')







state = st.sidebar.radio("State", ('X', 'Y'))
if st.sidebar.button('Submit'):
    st.write('')
    st.write('')
    st.write('You are at risk for heart disease')


sex = st.sidebar.radio("What is your sex", ('Male', 'Female'))

def sex_m(sex):
    if sex == 'Female':
        sex_m = 2
    else:
        sex_m = 1
    return sex_m

sex_v = sex_m(sex)

physical_activity = st.sidebar.slider('Rate how active you are ', 0, 5)

def physical_activity_m(physical_activity):
    physical_activity_m = physical_activity
    return physical_activity_m

physical_activity_v = physical_activity_m(physical_activity)





def yesno(var):
    if var == 'Yes':
        yesno = 1
    else:
        yesno = 2
    return yesno
cholesterol_high = st.sidebar.radio("High Cholesterol?", ('Yes', 'No')) #1 is yes #2 no
cholesterol_high_v = yesno(cholesterol_high)

stroke = st.sidebar.radio("Stroke?", ('Yes', 'No'))
stroke_v = yesno(stroke)


asthma = st.sidebar.radio("Asthma?", ('Yes', 'No'))
asthma_v = yesno(asthma)

cancer_skin = st.sidebar.radio("Skin Cancer?", ('Yes', 'No'))
cancer_skin_v = yesno(cancer_skin)

cancer = st.sidebar.radio("Any other type of cancer", ('Yes', 'No'))
cancer_other_v = yesno(cancer)

kidney_disease = st.sidebar.radio("Kidney Disease", ('Yes', 'No'))
kidney_disease_v = yesno(kidney_disease)

arthritis = st.sidebar.radio("Arthritis", ('Yes', 'No'))
arthritis_v = yesno(arthritis)

difficulty_walking = st.sidebar.radio("Difficulty Walking?", ('Yes', 'No'))
difficulty_walking_v = yesno(difficulty_walking)

smoker = st.sidebar.radio("Smoker", ('Yes', 'No'))
smoker_v = yesno(smoker)

alcoholic = st.sidebar.radio("alcoholic", ('Yes', 'No'))
alcoholic_v = yesno(alcoholic)

health_insurance = st.sidebar.radio("Health Insurance", ('Yes', 'No'))
health_insurance_v = yesno(health_insurance)


vaccine_flu = st.sidebar.radio("Do you recieve a Flu Vaccine", ('Yes', 'No'))
vaccine_flu_v = yesno(vaccine_flu)

vaccine_pneumonia = st.sidebar.radio("Do you recieve a pneumonia vaccine", ('Yes', 'No'))
vaccine_pneumonia_v = yesno(vaccine_pneumonia)

urban_status = st.sidebar.radio("Living", ('Urban', 'Rural'))

def urban_status_m(urban_status):
    if urban_status == 'Urban':
        urban_status_m = 1
    else:
        urban_status_m = 2
    return urban_status_m

urban_status_v = urban_status_m(urban_status)

ghealth = st.sidebar.slider('Rate your general health', 1, 5)

if ghealth == 1:
    general_health_fair = 0
    general_health_good = 0
    general_health_verygood = 0
    general_health_excellent = 1
elif ghealth == 2:
    general_health_fair = 0
    general_health_good = 0
    general_health_verygood = 1
    general_health_excellent = 0
elif ghealth == 3:
    general_health_fair = 0
    general_health_good = 1
    general_health_verygood = 0
    general_health_excellent = 0
elif ghealth == 4:
    general_health_fair = 1
    general_health_good = 0
    general_health_verygood = 0
    general_health_excellent = 0

age = st.sidebar.slider('What is your age?', 18, 80)

if age >= 80:
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
    age_12 =0
    age_13 = 1
elif age < 80 and age >= 74:
  
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
    age_12 =1
    age_13 = 0
elif age < 74 and age >= 70:
  
    age_2=0
    age_3=0
    age_4=0
    age_5=0
    age_6 =0
    age_7 =0
    age_8 =0
    age_9=0
    age_10 =0
    age_11 =1
    age_12 =0
    age_13 = 0
elif age < 69 and age >= 66:
  
    age_2=0
    age_3=0
    age_4=0
    age_5=0
    age_6 =0
    age_7 =0
    age_8 =0
    age_9=0
    age_10 =1
    age_11 =0
    age_12 = 0
    age_13 = 0

elif age < 66 and age >= 59:
  
    age_2=0
    age_3=0
    age_4=0
    age_5=0
    age_6 =0
    age_7 =0
    age_8 =0
    age_9=1
    age_10 =0
    age_11 =0
    age_12 = 0
    age_13 = 0
elif age < 60 and age >= 55:
  
    age_2=0
    age_3=0
    age_4=0
    age_5=0
    age_6 =0
    age_7 =0
    age_8 =1
    age_9=0
    age_10 =0
    age_11 =0
    age_12 = 0
    age_13 = 0
elif age < 55 and age >= 50:
  
    age_2=0
    age_3=0
    age_4=0
    age_5=0
    age_6 =0
    age_7 =1
    age_8 =0
    age_9=0
    age_10 =0
    age_11 =0
    age_12 = 0
    age_13 = 0
elif age < 54 and age >= 50:
  
    age_2=0
    age_3=0
    age_4=0
    age_5=0
    age_6 =1
    age_7 =0
    age_8 =0
    age_9=0
    age_10 =0
    age_11 =0
    age_12 = 0
    age_13 = 0

elif age < 50 and age >= 46:
  
    age_2=0
    age_3=0
    age_4=0
    age_5=1
    age_6 =0
    age_7 =0
    age_8 =0
    age_9=0
    age_10 =0
    age_11 =0
    age_12 = 0
    age_13 = 0

elif age < 46 and age >= 46:
  
    age_2=0
    age_3=0
    age_4=0
    age_5=1
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
    age_5=1
    age_6 =0
    age_7 =0
    age_8 =0
    age_9=0
    age_10 =0
    age_11 =0
    age_12 = 0
    age_13 = 0



income = st.sidebar.slider('Income in thousands', 1, 200)
if income < 100:


    income_2 = 0
    income_3 = 0
    income_4 = 1
    income_5 = 0 
    income_6 = 0
    income_7 = 0
    income_8 = 0
    income_9 = 0
    income_10 = 0
    income_11 = 0
else:
    income_2 = 0
    income_3 = 0
    income_4 = 0
    income_5 = 0 
    income_6 = 0
    income_7 = 0
    income_8 = 0
    income_9 = 1
    income_10 = 0
    income_11 = 0
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

race = st.sidebar.radio("Race", ('White', 'Black','Asian','American Indian/Alaskan Native', 'Other'))

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
elif race == 'Other':
    race_1 = 0
    race_2 = 0
    race_3 = 0
    race_4 = 1
    race_5 = 0
    race_6 = 0 
       
marital_status = st.sidebar.radio("Martial Status", ('Married', 'Divorce', 'Widowed','Seperated',' Never Married','Unmarried Couple'))

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


phealth = st.sidebar.slider('Number of days you were physcially unwell in last 30 days ', 0, 30)
phealth_v = phealth
mhealth = st.sidebar.slider('Number of days you were mentally unwell in last 30 days ', 0, 30)
mhealth_v = mhealth
bmi = st.sidebar.slider('BMI?', 12, 99)
bmi_v = bmi

alcohol_intake = st.sidebar.slider('How Many Drinks a Week', 0, 500)
alcohol_intake_v = alcohol_intake
fruit_intake = st.sidebar.slider('Fruit intake in times per day', 0, 99)
fruit_intake_v = fruit_intake

vegetable_intake = st.sidebar.slider('Dark vegetable intake in times per day', 0, 99)
vegetable_intake_v = vegetable_intake
french_fry_intake = st.sidebar.slider('French Fry intake in times per day', 0, 99)
french_fry_intake_v = french_fry_intake

education = st.sidebar.radio("Education Level", ('Did not graduate High School', 'High School', 'Attended College', 'Graduated College'))

if education == 'Did not graduate High School':

    education_1 = 1
    education_2 = 0
    education_3 = 0
    education_4 = 0
elif education == 'High School':

    education_1 = 0
    education_2 = 1
    education_3 = 0
    education_4 = 0
elif education == 'Attended College':

    education_1 = 0
    education_2 = 0
    education_3 = 1
    education_4 = 0
elif education == 'Graduate College':

    education_1 = 0
    education_2 = 0
    education_3 = 0
    education_4 = 1

household_child_count = st.sidebar.slider('Children in Household', 0, 88)
household_child_count_v = household_child_count


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

loaded_model = pickle.load(open('lgbm5.pkl', 'rb'))
result = loaded_model.predict(df)
prob = loaded_model.predict_proba
if result == 1:
    st.write("This Patient is at Risk For Heart Disease")
    st.write("Probability" + str(prob))
if result == 0:
    st.write("This Patient is NOT at Risk For Heart Disease")
    st.write("Probability" + str(prob))




















