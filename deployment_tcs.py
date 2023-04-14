# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:07:30 2023

@author: LENOVO
"""

import pandas as pd
import tensorflow
import keras
import streamlit as st
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from streamlit_option_menu import option_menu







model=tensorflow.keras.models.load_model("C:/Users/Siddharth Betadur/Documents/Tcs stock price/tcs_model2.h5")


def create_df(dataset,step):
    xxtrain,yytrain=[],[]
    for i in range(len(dataset)-step-1):
        a=dataset[i:(i+step),0]
        xxtrain.append(a)
        yytrain.append(dataset[i+step,0])
    return np.array(xxtrain),np.array(yytrain)




options = option_menu("Main Menu",["Home"], icons=['house','gear-fill',"envelope"], menu_icon="cast", default_index=0,orientation="horizontal")
st.title("STOCK MARKET FORECASTING")


a=st.sidebar.selectbox("STOCKS",("Select the stock","TCS"))


df=pd.read_csv("C:/Users/Siddharth Betadur/Documents/Tcs stock price/TCS.NS.csv")
st.dataframe(df)

# Plotting Close Pric
st.subheader("closing price")
fig3=plt.figure(figsize=(12,6))
plt.plot(df.Close)

# Plotting Close Price with MA100
st.subheader("closing price with 100MA")
ma1_100=df.Close.rolling(100).mean()
plt.plot(ma1_100)
st.pyplot(fig3)

df_=pd.read_csv("C:/Users/Siddharth Betadur/Documents/Tcs stock price/TCS.NS.csv")
df_=df_["Close"] 


# Performing LOG & SCALING 
df_log=np.log(df_)
normalizing=MinMaxScaler(feature_range=(0,1))
df_norm=normalizing.fit_transform(np.array(df_log).reshape(-1,1))


t_s=100                            
df_x,df_y=create_df(df_norm, t_s)                                
fut_inp=df_y[2267:]
fut_inp=fut_inp.reshape(1,-1)   
temp_inp=list(fut_inp) 
temp_inp=temp_inp[0].tolist()


lst_out=[]   
n_steps=100
i=1



int_val = st.number_input('day', min_value=1, max_value=100, value=5, step=1)
while(i<int_val):
        if(len(temp_inp)>100):
            fut_inp=np.array(temp_inp[1:])
            fut_inp=fut_inp.reshape(1,-1)
            fut_inp=fut_inp.reshape((1,n_steps,1))
            yhat=model.predict(fut_inp,verbose=0)
            temp_inp.extend(yhat[0].tolist())
            temp_inp=temp_inp[1:]
            lst_out.extend(yhat.tolist())
            i=i+1
        else:
            fut_inp
            yhat=model.predict(fut_inp,verbose=0)
            temp_inp.extend(yhat[0].tolist())
            lst_out.extend(yhat.tolist())
            i=i+1


lst_out=normalizing.inverse_transform(lst_out)
lst_out=np.exp(lst_out) 
st.dataframe(lst_out)


c=lst_out
fig4=plt.figure(figsize=(12,6))
plt.plot(c)
st.pyplot(fig4)