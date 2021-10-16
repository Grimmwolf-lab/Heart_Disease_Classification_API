import pickle as pkl
import pandas as pd
from sklearn.metrics import accuracy_score

test = pd.read_csv('C:/Users/kaila/Desktop/Heart_Diseases_API/data/test.csv')
predict_on= test.drop('target',axis=1)

true_predict = test['target']



model = pkl.load(open('C:/Users/kaila/Desktop/Heart_Diseases_API/model/heart_disease.pkl','rb'))

predictions = model.predict(predict_on)
#print(predictions)


#my_predictions = pd.DataFrame({'Id':test.id , 'target': predictions})

#my_predictions.to_csv(r'C:/Users/kaila/Desktop/Heart_Diseases_API/data/predictions.csv', index=False)
print(accuracy_score(true_predict,predictions))
print(model.predict_proba([[56,0,0,200,288,1,0,133,1,4,0,2,3]]))