# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:02:43 2022

@author: Kevin Boss
"""
from PIL import Image
#from streamlit_shap import st_shap
import streamlit as st
import numpy as np 
import pandas as pd 
import time
import plotly.express as px 
import seaborn as sns
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error,confusion_matrix,accuracy_score,recall_score,precision_score,classification_report,roc_auc_score
import shap
import catboost
from catboost import CatBoostClassifier
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

plt.style.use('default')

st.set_page_config(
    page_title = 'Real-Time Fraud Detection',
    page_icon = '🕵️‍♀️',
    layout = 'wide'
)

# dashboard title
#st.title("Real-Time Fraud Detection Dashboard")
st.markdown("<h1 style='text-align: center; color: black;'>机器学习： 实时识别出虚假销售</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>Real-Time Fraud Detection</h1>", unsafe_allow_html=True)

# side-bar 
def user_input_features():
    st.sidebar.header('Make a prediction')
    st.sidebar.write('User input parameters below ⬇️')
    a1 = st.sidebar.slider('Action1', -31.0, 3.0, 0.0)
    a2 = st.sidebar.slider('Action2', -5.0, 13.0, 0.0)
    a3 = st.sidebar.slider('Action3', -20.0, 6.0, 0.0)
    a4 = st.sidebar.slider('Action4', -26.0, 7.0, 0.0)
    a5 = st.sidebar.slider('Action5', -4.0, 5.0, 0.0)
    a6 = st.sidebar.slider('Action6', -8.0, 4.0, 0.0)
    a7 = st.sidebar.slider('Sales Amount', 1.0, 5000.0, 1000.0)
    a8 = st.sidebar.selectbox("Gender?", ('Male', 'Female'))
    a9 = st.sidebar.selectbox("Agent Status?", ('Happy','Sad','Normal'))
    
    output = [a1,a2,a3,a4,a5,a6,a7,a8,a9]
    return output

outputdf = user_input_features()



# understand the dataset
df = pd.read_excel('fraud.xlsx')


st.title('Dataset')
if st.button('View some random data'):
    st.write(df.sample(5))
    
st.write(f'The dataset is trained on Catboost with totally length of: {len(df)}. 0️⃣ means its a real transaction, 1️⃣ means its a Fraud transaction. data is unbalanced (not⚖️)')


unbalancedf = pd.DataFrame(df.Class.value_counts())
st.write(unbalancedf)



# 需要一个count plot
placeholder = st.empty()
placeholder2 = st.empty()
placeholder3 = st.empty()

with placeholder.container():
    f1,f2,f3 = st.columns(3)

    with f1:
        a11 = df[df['Class'] == 1]['Action1']
        a10 = df[df['Class'] == 0]['Action1']
        hist_data = [a11, a10]
        #group_labels = ['Real', 'Fake']
        fig = ff.create_distplot(hist_data,group_labels = ['Fraud', 'legit'])
        fig.update_layout(title_text='Action 1')
        st.plotly_chart(fig, use_container_width=True)
    with f2:
        a21 = df[df['Class'] == 1]['Action2']
        a20 = df[df['Class'] == 0]['Action2']
        hist_data = [a21, a20]
        #group_labels = ['Real', 'Fake']
        fig = ff.create_distplot(hist_data,group_labels = ['Fraud', 'real'])
        fig.update_layout(title_text='Action 2')
        st.plotly_chart(fig, use_container_width=True)
    with f3:
        a31 = df[df['Class'] == 1]['Action3']
        a30 = df[df['Class'] == 0]['Action3']
        hist_data = [a31, a30]
        #group_labels = []
        fig = ff.create_distplot(hist_data, group_labels = ['Fraud', 'real'])
        fig.update_layout(title_text='Action 3')
        st.plotly_chart(fig, use_container_width=True)

with placeholder2.container():
    f1,f2,f3 = st.columns(3)

    with f1:
        a41 = df[df['Class'] == 1]['Action4']
        a40 = df[df['Class'] == 0]['Action4']
        hist_data = [a41, a40]
        #group_labels = ['Real', 'Fake']
        fig = ff.create_distplot(hist_data, group_labels = ['Fraud', 'real'])
        fig.update_layout(title_text='Action 4')
        st.plotly_chart(fig, use_container_width=True)
    with f2:
        a51 = df[df['Class'] == 1]['Action5']
        a50 = df[df['Class'] == 0]['Action5']
        hist_data = [a51, a50]
        #group_labels = ['Real', 'Fake']
        fig = ff.create_distplot(hist_data, group_labels = ['Fraud', 'real'])
        fig.update_layout(title_text='Action 5')
        st.plotly_chart(fig, use_container_width=True)
    with f3:
        a61 = df[df['Class'] == 1]['Action6']
        a60 = df[df['Class'] == 0]['Action6']
        hist_data = [a61, a60]
        #group_labels = ['Real', 'Fake']
        fig = ff.create_distplot(hist_data, group_labels = ['Fraud', 'real'])
        fig.update_layout(title_text='Action 6')
        st.plotly_chart(fig, use_container_width=True)



df2 = df[['Class','Gender']].value_counts().reset_index()

df3 = df[['Class','Agent status']].value_counts().reset_index()



with placeholder3.container():
    f1,f2,f3 = st.columns(3)

    with f1:
        as1 = df[df['Class'] == 1]['Sales Amount']
        as0 = df[df['Class'] == 0]['Sales Amount']
        hist_data = [as1, as0]
        #group_labels = ['1', '0']
        colors = [px.colors.qualitative.Plotly[0], px.colors.qualitative.Plotly[1]]
        fig = ff.create_distplot(hist_data, colors=colors, group_labels = ['Fraud', 'real'])
        fig.update_layout(title_text='Sales Amount')
        st.plotly_chart(fig, use_container_width=True)
    with f2:
        #fig = plt.figure()
        fig = px.bar(df2, x='Class', y=0, color='Gender', color_continuous_scale=px.colors.qualitative.Plotly,  title=" Gender: 🔴Female; 🔵Male")
        st.write(fig)        


    with f3:
        fig = px.bar(df3, x='Class', y= 0, color="Agent status", title="Agent status:  💚sad; ❤️happy; 💙normal")
        st.write(fig)   







st.title('SHAP Value')

image4 = Image.open('summary.png')
shapdatadf =pd.read_excel(r'shapdatadf.xlsx')
shapvaluedf =pd.read_excel(r'shapvaluedf.xlsx')





placeholder5 = st.empty()
with placeholder5.container():
    f1,f2 = st.columns(2)

    with f1:
        st.subheader('Summary plot')
        st.write('👈 class 0: Real')
        st.write('👉 class 1: Fraud')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.image(image4)     
    with f2:
        st.subheader('Dependence plot for features')
        cf = st.selectbox("Choose a feature", (shapdatadf.columns))
        

        fig = px.scatter(x = shapdatadf[cf], 
                         y = shapvaluedf[cf], 
                         color=shapdatadf[cf],
                         color_continuous_scale= ['blue','red'],
                         labels={'x':'Original value', 'y':'shap value'})
        st.write(fig)  

catmodel = CatBoostClassifier()
catmodel.load_model("fraud")

st.title('Make predictions in real time')
outputdf = pd.DataFrame([outputdf], columns= shapdatadf.columns)

#st.write('User input parameters below ⬇️')
#st.write(outputdf)




p1 = catmodel.predict(outputdf)[0]
p2 = catmodel.predict_proba(outputdf)


placeholder6 = st.empty()
with placeholder6.container():
    f1,f2 = st.columns(2)
    with f1:
        st.write('User input parameters below ⬇️')
        st.write(outputdf)
        st.write(f'Predicted class: {p1}')
        st.write('Predicted class Probability')
        st.write('0️⃣ means its a real transaction, 1️⃣ means its a Fraud transaction')
        st.write(p2)
    with f2:
        
        explainer = shap.Explainer(catmodel)
        shap_values = explainer(outputdf)

        #st_shap(shap.plots.waterfall(shap_values[0]),  height=500, width=1700)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.plots.waterfall(shap_values[0])
        st.pyplot(bbox_inches='tight')

