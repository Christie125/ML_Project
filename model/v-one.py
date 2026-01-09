#Following Google's linear regression tutorial will help me to access some guidance on how to create my model successfully
#Thus, I have decided to use the some of the same libraries as used in the tutorial, after having researched them for a deeper understanding of their functionality
#I'll also use sci-kit learn

#__Importing all necessary libraries__

#This library allows me to work with numbers and arrays nicely, but cruicially is also neccessary for the pandas library to work
#I'm not sure exactly what I will do with just vanilla numpy, but as pandas depends on it, I have imported it here
import numpy as np
#This is pandas -- it's basically like working with a spreadsheet like Google Sheets in Python, which is obviously important to machine learning as the it relies on data and dataframes are a nice way to structure this data so its easy to work with and understand
import pandas as pd
#These libraries will allow me to generate visuals as I make my model, which is good for me as I find that I understand difficult concepts better through visuals
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#This library actually does the machine learning stuff
from sklearn.linear_model import LinearRegression
#This library will help me to evaluate how well my model performs by showing me the average loss
from sklearn.metrics import mean_absolute_error
#This library will help me to split my data into training and testing sets, which is important for evaluating how well my model performs on unseen data
from sklearn.model_selection import train_test_split
#This, as will be discussed later, will help me to normalise my data
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv("model/ml_data.csv")
#print(dataset)
print(dataset.describe(include='all'))
print(dataset.info())
#Study hours shows the most correlation with scores. This can be explained as the more one studies, the better one knows the material, and by extension the better one performs in an exam.
print(dataset.corr())
#Visualisation of correlation matrix
corr = px.scatter_matrix(dataset, dimensions=["study_hours", "exam_score", "age", "sleep_hours", "class_attendance", "exam_difficulty"], title="Scatter Matrix")
print("Creating matrix plot...") # I hate waiting ages not knowing what my code is doing
#corr.show()

#Defining the x and y of the graph
#x is the independent variables (the features that will be used to predict the exam score) -- they will be the inputs
#y is the dependent variable (the target that we want to predict) -- it will be the output
#this can easily be visualised on a graph where x is on the horizontal axis and y is on the vertical axis
#the reshaping is requred to make the exam_score a 2D array instead of a 1D array, as required by sci-kit learn
x = np.array(dataset[['study_hours', 'age', 'sleep_hours', 'class_attendance', 'exam_difficulty']])
y = np.array(dataset['exam_score']).reshape(-1, 1)

#Splitting the data into training and testing sets
print("Splitting data into training and testing sets...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=25)

print("Training and normalising model...")
#Normalising the data to improve model performance
scaler = StandardScaler()
scaler.fit_transform(x_train)
#Creating the model using the built-in Linear Regression model from sci-kit learn
model = LinearRegression()
#Make the model learn the relationship between x and y using the training data
scaler.transform(x_test)
model.fit(x_train, y_train)
#Seeing what coefficients and bias the model has learned
print("Coefficients:", model.coef_)
print("Bias:", model.intercept_)
#I can see that study_hours has the highest coefficient, which makes sense as it had the highest correlation with exam_score in the correlation matrix
#I can see that my model has already been trained! Using libraries makes things so much easier :)
#Also the bias looks legit

#Test with random numbers to see if it works
test = model.predict([[5, 17, 8, 90, 2]])
print("Predicted exam score for a student who studies 5 hours, is 17 years old, sleeps 8 hours, has 90% class attendance, and finds the exam difficulty to be 2/3:", test)
#Ok yay I think it works!
#Time to actually evaluate the model now to see if its any good at actually predicting exam scores

#Evaluating the model using the testing data
y_pred = model.predict(x_test)
#Comparing the predicted exam scores with the actual exam scores
mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)

#Visualising the results vs predictions
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Actual vs Predicted Exam Scores")
plt.show()

#Oh no, the MAE is quite high -- around 9.3!
#This means that on average, the model's predictions are off by 9.3 points
#Next steps: training the model differently to improve it

#I need to try various stratergies to improve the model's performance. 
#The numbers vary quite a lot so normalisation may help. To do this, I will use StandardScaler from sklearn.preprocessing to standardise the features by removing the mean and scaling to unit variance.
#That doesn't seem to have worked -- MAE is still high.
#I will try to add the coloums I cleaned off the data earlier back in to see if they help improve the model
#I will do this in v2 of the model
