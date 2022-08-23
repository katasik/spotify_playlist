#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#import csv file
df = pd.read_csv (r'data/top_100.csv')

print(df.head())

#let's plot the tempo column

plt.figure(figsize= (10,7))
x1 = df["Tempo"]
plt.hist(x1, bins = 20, color = 'green')
plt.title("Tempo level of 100 favorite work songs")
plt.xlabel("Tempo level of songs")
plt.ylabel("Number of songs")
plt.savefig('machine_learning_plots/tempo_lev.png')
plt.show()


#let's plot the danceability column

plt.figure(figsize= (10,7))
x2 = df["Danceability"]
plt.hist(x2, bins = 20, color = 'lightblue')
plt.title("Danceability level of 100 favorite work songs")
plt.xlabel("Danceability level")
plt.ylabel("Number of songs")
plt.savefig('machine_learning_plots/dance_lev.png')
plt.show()

#let's see if danceability and tempo are related with a linear regression
#As both columns are roughly normally distributed, a simple linear regression likely will be appropriate
#but we will need to test the distribution of the residuals

lr_df = df[["Tempo", "Danceability"]]

#let's see the relationship between the two variables
sns.pairplot(lr_df)
plt.savefig('machine_learning_plots/pairwise_plot.png')
plt.show()

#Assign values in x and y

#tempo
x = lr_df.iloc[:, :-1].values
#danceability
y = lr_df.iloc[:, -1].values

print(x)
print(y)


#we need to create a train and a test dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .33, random_state = 42)

#let's see the created dataframes
print(x_test)
print(x_train)
print(y_test)
print(y_train)

#using the train set to train the model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

#let's validate the model with the test set
y_pred = lr.predict(x_test)
print(y_pred)

#plotting it on a graph
plt.scatter(x_train, y_train, color = 'green')
plt.plot(x_train, lr.predict(x_train), color = 'black')
plt.title('Training Set')
plt.xlabel('Tempo level of songs')
plt.ylabel('Danceability level of songs')
plt.savefig('machine_learning_plots/training_set_regline.png')
plt.show()

#we can see a negative relationship between the songs' danceability level and tempo level
plt.scatter(x_test, y_test, color = 'black')
plt.plot(x_train, lr.predict(x_train), color = 'blue')
plt.title('Test Set')
plt.xlabel('Tempo level of songs')
plt.ylabel('Danceability level of songs')
plt.savefig('machine_learning_plots/test_set_regline.png')
plt.show()

#in the test set, there is also a negative relationship between the tempo and danceability levels

#Let's see what is the stope and intercept of the model
#slope
b = lr.coef_
print("Coefficient  :", b)

#intercept
a = lr.intercept_
print("Intercept : ", a)

#This means that for every 1 unit increase in tempo level, the danceability level decreases by 0.002

#For instance, let's check the mean tempo level of my favorite songs
print(lr_df.describe())

#It is 120.77
#We can calculate the danceability level of my songs at the tempo level of 120.77
print(lr.predict([[120.77]]))
#It is 0.597, thus in general, based on the average tempo level of my favorite songs, I will
#probably prefer listening to more danceable songs (>0.6) in the future as well

#Let's see the model's fit with the MSE (Mean Squared Error)
from sklearn import metrics
print('Mean Squared Error (MSE   :', metrics.mean_squared_error(y_test, y_pred))

#The mean square error is 0.025, which is pretty low, reflecting that the
# predicted values relatively closely matched the expected values

#Let's check the OLS summary
import statsmodels.api as sm

x_stat = sm.add_constant(x_train)
regsummary = sm.OLS(y_train, x_stat).fit()
print(regsummary.summary())

#We can see that the R2 value is 0.15, which is not too high. This just means
#that the tempo level of the songs explains onyl 15% of the variance
# of the songs' danceability level
