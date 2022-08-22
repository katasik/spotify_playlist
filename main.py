#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import csv file

df = pd.read_csv (r'data/top_100.csv')
print (df)

plt.hist(df(8), bins=10)
plt.show()