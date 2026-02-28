#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline 
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib
import streamlit as st


# In[30]:


df = pd.read_csv("C:\\Users\\USER-11\\Downloads\\kc_house_data.csv (1)\\kc_house_data.csv")


# In[31]:


df.shape


# In[32]:


df.head()


# In[33]:


df['condition'].mean()


# In[34]:


df.isnull().sum()


# In[35]:


df.dtypes


# In[36]:


plt.figure(figsize=(10, 6))
bedroom_counts = df['bedrooms'].value_counts().sort_index()
sns.barplot(x=bedroom_counts.index, y=bedroom_counts.values, color='skyblue')
plt.title('Count of Houses by Number of Bedrooms', fontsize=14)
plt.xlabel('Bedrooms', fontsize=12)
plt.ylabel('Number of Houses', fontsize=12)
plt.savefig('visual_bedroom_count.png')
plt.show()


# In[37]:


plt.figure(figsize=(10, 6))
mean_price_bed = df.groupby('bedrooms')['price'].mean().sort_index()
sns.barplot(x=mean_price_bed.index, y=mean_price_bed.values, color='salmon')
plt.title('Average House Price by Number of Bedrooms', fontsize=14)
plt.xlabel('Bedrooms', fontsize=12)
plt.ylabel('Average Price (USD)', fontsize=12)
plt.savefig('visual_price_bedroom.png')
plt.show()


# In[38]:


plt.figure(figsize=(10, 6))
mean_price_grade = df.groupby('grade')['price'].mean().sort_index()
plt.plot(mean_price_grade.index, mean_price_grade.values, marker='o', linestyle='-', color='teal', linewidth=2)
plt.title('Average House Price by Building Grade', fontsize=14)
plt.xlabel('Grade', fontsize=12)
plt.ylabel('Average Price (USD)', fontsize=12)
plt.grid(True)
plt.savefig('visual_price_grade.png')
plt.show()


# In[39]:


numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of House Features', fontsize=16)
plt.savefig('Correlation Heatmap of House Features')
plt.show()


# In[40]:


x = df.drop(['id','date','floors','waterfront','view','sqft_lot','sqft_above','sqft_basement','yr_renovated','sqft_living15','sqft_lot15','price'],axis=1)
y = df['price']


# In[41]:


numerical_cols = x.select_dtypes(include=['int64','float64']).columns.tolist()


# In[42]:


categorical_cols = x.select_dtypes(include=['object']).columns.tolist()


# In[43]:


numerical_transformer= Pipeline (steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])


# In[44]:


categorical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])


# In[45]:


preprocessor =ColumnTransformer(transformers=[
    ('num',numerical_transformer,numerical_cols ),
    ('cat',categorical_transformer,categorical_cols )
])


# In[46]:


X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[47]:


model =Pipeline(steps=[
    ('pre',preprocessor),('reg',LinearRegression())
])


# In[48]:


model.fit(X_train,y_train)


# In[49]:


y_pred = model.predict(X_test)

print(f'Accuracy:{r2_score(y_pred,y_test)*100:.2f}')


# In[50]:


model2 = Pipeline(steps=[
    ('pre',preprocessor),('reg',RandomForestRegressor(n_estimators=200,random_state=42))
])


# In[51]:


model2.fit(X_train,y_train)


# In[52]:


y_pred2 =model2.predict(X_test)

print(f'Accuracy:{r2_score(y_pred2,y_test)*100:.2f}')


# In[53]:


joblib.dump(model2,'randomforestregressor.pkl')


# In[54]:


load=joblib.load('randomforestregressor.pkl')
st.title('House Price Prediction')
bedrooms=st.number_input('bedrooms')
bathrooms=st.number_input('bathrooms')
sqft_living=st.number_input('sqft_living')
grade=st.number_input('grade')
condition=st.number_input('condition')
yr_built=st.number_input('yr_built')
zipcode=st.number_input('zipcode')
lat=st.number_input('lat')
long =st.number_input('long')
if st.button('predict'):
    data = pd.DataFrame({
         'bedrooms':[bedrooms],
        'bathrooms':[bathrooms],
      'sqft_living':[sqft_living],
            'grade':[grade],
        'condition':[condition],
         'yr_built':[yr_built],
          'zipcode':[zipcode],
              'lat':[lat],
             'long':[long]
    })
    prediction =load.predict(data)
    st.success(f'Price:{prediction[0]}')


# In[ ]:




