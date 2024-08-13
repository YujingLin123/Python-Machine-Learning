import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

music_data = pd.read_csv('music.csv')   # control+/ 同时注释多行
X = music_data.drop(columns=['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(X, y)

joblib.dump(model, 'music-recommender.joblib')

#predictions = model.predict(X_test)


## load the model and run it for making predictions
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

# music_data = pd.read_csv('music.csv')
# X = music_data.drop(columns=['genre'])
# y = music_data['genre']

# model = DecisionTreeClassifier()
# model.fit(X, y)

model = joblib.load( 'music-recommender.joblib')
predictions = model.predict([ [21, 1] ])
predictions

