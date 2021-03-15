# Import required libraries
import pandas as pd
from numpy import mean, std
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Load the player_dataset
player_dataset = pd.read_csv('final_player_dataset.csv')

# Separate out independent and dependent variables
X = player_dataset.iloc[:, :-1].values
y = player_dataset.iloc[:, -1].values 

# scaling the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Define evaluation model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# Defining the model
model = GradientBoostingRegressor(learning_rate=0.1, 
                                  n_estimators=60,
                                  max_depth=5,
                                  min_samples_split=2,
                                  min_samples_leaf=1,
                                  max_features = 2,
                                  subsample=0.8,
                                  random_state=10)
model.fit(X,y)                                  

# evaluating performance of the model
score = cross_val_score(model, X, y, scoring='r2', cv=cv, n_jobs=-1)
print(score.mean())


# Saving the model
pickle.dump(model, open('gbr_model.pkl','wb'))
pickle.dump(scaler, open('scaler.pkl','wb'))


							