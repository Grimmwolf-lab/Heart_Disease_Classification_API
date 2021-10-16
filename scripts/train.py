# Importing modules
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import pickle as pkl

#Setup random seed
np.random.seed(42)


test = pd.read_csv('C:/Users/kaila/Desktop/Heart_Diseases_API/data/train.csv')

# Since the dataset is already cleaned and categorical features are handled already 
# we only need to do hyperparameter tunning.
model = Pipeline(steps=[('model', RandomForestClassifier())])


X = test.drop('target', axis=1)
y = test['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# We will be using gridsearchCV
# pipe_grid = {
#             "model__n_estimators":[10,100,150,200],
#             "model__max_depth":[None,5,10],
#             "model__max_features":["auto"],
#             "model__min_samples_split":[2,4]
# }

#gs_model = GridSearchCV(model, pipe_grid, cv=5, verbose=2)
model.fit(X_train,y_train)

print(model.score(X_test, y_test))

pkl.dump(model, open('C:/Users/kaila/Desktop/Heart_Diseases_API/model/heart_disease.pkl','wb'))



