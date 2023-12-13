import streamlit as st
import pickle
import numpy as np
import pandas as pd



st.title('Welcome to Ecomers ')
def load_model():
    with open('lmmodel.pkl','rb') as file:
        data=pickle.load(file)
    return data

data=load_model()
l_model=data['model']

st.write("Fill the below requirments to predict")

sess=st.number_input(label="Avg. Session Length")

app=st.number_input(label="Time on App")

web=st.number_input(label="Time on Website")

membership=st.number_input(label="Length of Membership")

x=np.array([[sess,app,web,membership]])
Yearly_Amount_Spent=l_model.predict(x)
pred=st.button("Yearly Amount Spent")
if pred:
    st.subheader(f'Yearly Amount Spent is {Yearly_Amount_Spent[0]}')

