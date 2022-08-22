#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#import csv file
df = pd.read_csv (r'data/top_100.csv')

#first 10 rows of the dataset
print(df.head(10))

#let's see descriptive stats of the columns
print(df.describe())

#what data types do we have?
print(df.info())

#let's plot the popularity column

plt.figure(figsize= (10,7))
x1 = df["Popularity"]
plt.hist(x1, bins = 20, color = 'green')
plt.title("Popularity level of 100 favorite work songs")
plt.xlabel("Popularity level (on a scale from 0 to 100)")
plt.ylabel("Number of songs")
plt.show()

#let's plot the danceability column

plt.figure(figsize= (10,7))
x2 = df["Danceability"]
plt.hist(x2, bins = 20, color = 'lightblue')
plt.title("Danceability level of 100 favorite work songs")
plt.xlabel("Danceability level (on a scale from 0 to 1)")
plt.ylabel("Number of songs")
plt.show()

#Let's see the relationship between them

#first use the information of the two important columns in our case
scatter_df = print(df[["Popularity", "Danceability"]].info())
#create x and y variables
x = df["Danceability"]
y = df["Popularity"]

#plot them on a scatterplot
plt.scatter(x, y)
plt.title("Danceability vs. Popularity")
plt.xlabel("Danceability level")
plt.ylabel("Popularity level")
plt.show()

#let's also plot a regression line
ax = sns.regplot(x=x, y=y)
plt.title("Danceability vs. Popularity")
plt.xlabel("Danceability level")
plt.ylabel("Popularity level")
plt.show()

#There is no correlation between the songs' popularity and danceability levels!