from models import BodyCheck
from fastapi import FastAPI
import uvicorn
from models import BodyCheck
import pickle as pkl


app = FastAPI()

model = pkl.load(open('heart_disease.pkl','rb'))





@app.get('/')
def index():
    return {'message':'Hello World!'}

@app.get('/{name}')
def get_name(name: str):
    return {'message': f'Hello {name}'}



@app.post('/{predict}')
def predict_disease(data:BodyCheck):
    data = data.dict()
    age = data['age']
    sex = data['sex']
    chest_pain = data['cp']
    Resting_blood_pressure = data['trestbps']
    cholestrol = data['chol']
    fasting_blood_sugar = data['fbs']
    restecg = data['restecg'] #resting electrocradiographic result
    thalach = data['thalach'] #maximum heart rate achieved
    exang = data['exang'] #exercise induced angina
    oldpeak = data['oldpeak'] # ST depression induced by exercise realtive to rest
    slope = data['slope'] # Slope of the peak exercise ST segment
    ca = data['ca'] #Number of major vessels coloured by flourosopy
    thal = data['thal'] #Thalassemia (hemoglobin level inherited by parents)
    print(model.predict([[age, sex, chest_pain,Resting_blood_pressure,cholestrol,fasting_blood_sugar,restecg,thalach,exang,oldpeak,slope,ca,thal]]))
    prediction = model.predict([[age, sex, chest_pain,Resting_blood_pressure,cholestrol,fasting_blood_sugar,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    if prediction[0]>0.6:
        return f'Patient has probability to have a heart disease.'
    else:
        return f'Patient is safe from any kind of heart disease problem.'


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port = 8000)

