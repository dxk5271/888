import streamlit as st
import pandas as pd
import time


st.title("Hello")

option = st.selectbox(
     'Select a Model',  
    ('Neural Network', 'Logistic Regression', 'KNN'))

if st.button('Run Model'):
    with st.spinner('Wait for it...'):
        train = pd.read_parquet('ori_test_0218.parquet')
        st.write('here')
        #test = pd.read_parquet(ori_test_path)




   


