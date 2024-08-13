# Python-Machine-Learning

## Process
1. Import the Data
2. Clean the Data
3. Split the Data into Training
4. Create a Model
5. Train the Model
6. Make Predictions
7. Evaluate and improve

## Libraries and Tools
1. numpy provides multi-dimensional array.
2. pandas which is a data analysis library that provides a concept called data frame.
3. matplotlib which is a two-dimensional plotting library for creating graphs and plots.
4. scikit-learn which is one of the most popular maching learning libraries that provides all these common algorithms like decision trees neural networks and so on.

## Dataset
https://www.kaggle.com/datasets/gregorut/videogamesales?resource=download

## Preparing the data
import pandas as pd
music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']
y


