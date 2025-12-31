#Following Google's linear regression tutorial will help me to access some guidance on how to create my model successfully
#Thus, I have decided to use the same libraries as used in the tutorial, after having researched them for a deeper understanding of their functionality

#__Importing all necessary libraries__

#This library allows me to work with numbers and arrays nicely, but cruicially is also neccessary for the pandas library to work
#I'm not sure exactly what I will do with just vanilla numpy, but as pandas depends on it, I have imported it here
import numpy as np
#This is pandas -- it's basically like working with a spreadsheet like Google Sheets in Python, which is obviously important to machine learning as the it relies on data and dataframes are a nice way to structure this data so its easy to work with and understand
import pandas as pd
#This library will allow me to generate visuals as I make my model, which is good for me as I find that I understand difficult concepts better through visuals
import matplotlib.pyplot as plt
#This library does some complictaed machine learning stuff, making it easy for me to create my model
import keras
#this library abstracts things into nice understandable concepts for me to use.

#REMINDER TO INSTALL TENSORFLOW AND GOOGLE-ML-EDU

dataset = pd.read_csv("model/ml_data.csv")
#print(dataset)
dataset.describe(include='all')