import pandas as pd
import numpy as np

# Get in that file
wine_train_path = 'ai_club_wine_comp\\train.csv'

# Load DataFrame
train_wine_data = pd.read_csv(wine_train_path)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
#from sklearn.model_selection import train_test_split

wine_target = train_wine_data.quality
wine_predictors = train_wine_data.drop(['quality', 'chlorides'], axis=1)

#X_train, X_test, y_train, y_test = train_test_split(wine_predictors, wine_target, train_size=0.7, test_size=0.3)

model = RandomForestRegressor(n_estimators=100)
model.fit(wine_predictors, wine_target)
#preds = model.predict(X_test)
#print(mean_absolute_error(y_test, preds))


wine_test_path = 'ai_club_wine_comp\\test.csv'
test_wine_data = pd.read_csv(wine_test_path)
test_predictors = test_wine_data.drop(['id', 'chlorides'], axis=1)
predictions = model.predict(test_predictors)
print(predictions)
predictions = pd.DataFrame(predictions, columns=['quality']).to_csv(
    'ai_club_wine_comp\\jack_dubbs_predictions.csv')

'''
6.6	0.16	0.4	   1.5	0.044	48	143	0.9912	3.54	0.52	12.4	7
6.6	0.17	0.38	1.5	0.032	28	112	0.9914	3.25	0.55	11.4	7
6.2	0.66	0.48	1.2	0.029	29	75	0.9892	3.33	0.39	12.8	8
6.2	0.66	0.48	1.2	0.029	29	75	0.9892	3.33	0.39	12.8	8
6.8	0.26	0.42	1.7	0.049	41	122	0.993	3.47	0.48	10.5	8
7.1	0.32	0.32	11	0.038	16	66	0.9937	3.24	0.4	    11.5    3
'''
