#!/usr/bin/env python
# coding: utf-8

# In[96]:


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


# In[97]:


df = pd.read_csv("kc_house_data.csv")


# In[98]:


df.shape


# In[99]:


df.head()


# In[100]:


df['condition'].mean()


# In[101]:


df.isnull().sum()


# In[102]:


df.dtypes


# In[103]:


plt.figure(figsize=(10, 6))
bedroom_counts = df['bedrooms'].value_counts().sort_index()
sns.barplot(x=bedroom_counts.index, y=bedroom_counts.values, color='skyblue')
plt.title('Count of Houses by Number of Bedrooms', fontsize=14)
plt.xlabel('Bedrooms', fontsize=12)
plt.ylabel('Number of Houses', fontsize=12)
plt.savefig('visual_bedroom_count.png')
plt.show()


# In[104]:


plt.figure(figsize=(10, 6))
mean_price_bed = df.groupby('bedrooms')['price'].mean().sort_index()
sns.barplot(x=mean_price_bed.index, y=mean_price_bed.values, color='salmon')
plt.title('Average House Price by Number of Bedrooms', fontsize=14)
plt.xlabel('Bedrooms', fontsize=12)
plt.ylabel('Average Price (USD)', fontsize=12)
plt.savefig('visual_price_bedroom.png')
plt.show()


# In[105]:


plt.figure(figsize=(10, 6))
mean_price_grade = df.groupby('grade')['price'].mean().sort_index()
plt.plot(mean_price_grade.index, mean_price_grade.values, marker='o', linestyle='-', color='teal', linewidth=2)
plt.title('Average House Price by Building Grade', fontsize=14)
plt.xlabel('Grade', fontsize=12)
plt.ylabel('Average Price (USD)', fontsize=12)
plt.grid(True)
plt.savefig('visual_price_grade.png')
plt.show()


# In[106]:


numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of House Features', fontsize=16)
plt.savefig('Correlation Heatmap of House Features')
plt.show()


# In[107]:


x = df.drop(['id','date','floors','waterfront','view','sqft_lot','sqft_above','sqft_basement','yr_renovated','sqft_living15','sqft_lot15','price'],axis=1)
y = df['price']


# In[108]:


numerical_cols = x.select_dtypes(include=['int64','float64']).columns.tolist()


# In[109]:


categorical_cols = x.select_dtypes(include=['object']).columns.tolist()


# In[110]:


numerical_transformer= Pipeline (steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaler',StandardScaler())
])


# In[111]:


categorical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])


# In[112]:


preprocessor =ColumnTransformer(transformers=[
    ('num',numerical_transformer,numerical_cols ),
    ('cat',categorical_transformer,categorical_cols )
])


# In[113]:


X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[114]:


model =Pipeline(steps=[
    ('pre',preprocessor),('reg',LinearRegression())
])


# In[115]:


model.fit(X_train,y_train)


# In[116]:


y_pred = model.predict(X_test)

print(f'Accuracy:{r2_score(y_pred,y_test)*100:.2f}')


# In[117]:


model2 = Pipeline(steps=[
    ('pre',preprocessor),('reg',RandomForestRegressor(n_estimators=200,random_state=42))
])


# In[118]:


model2.fit(X_train,y_train)


# In[119]:


y_pred2 =model2.predict(X_test)

print(f'Accuracy:{r2_score(y_pred2,y_test)*100:.2f}')


# In[120]:


joblib.dump(model2,'randomforestregressor.pkl')


# In[121]:


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
        


# In[123]:


# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Load the trained model
model = joblib.load('randomforestregressor.pkl')

# Initialize FastAPI
app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices based on features",
    version="1.0"
)

# Allow CORS so that frontend (like Streamlit or React) can access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input schema
class HouseFeatures(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    grade: int
    condition: int
    yr_built: int
    zipcode: int
    lat: float
    long: float

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to House Price Prediction API!"}

# Prediction endpoint
@app.post("/predict")
def predict_price(features: HouseFeatures):
    # Convert input to DataFrame
    data = pd.DataFrame([features.dict()])
    
    # Make prediction
    prediction = model.predict(data)
    
    # Return prediction
    return {"predicted_price": float(prediction[0])}


# In[ ]:





# In[ ]:




