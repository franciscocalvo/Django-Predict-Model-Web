from django.template import Template,Context,loader
from datetime import datetime
from django.http import HttpResponse
from django.template.loader import get_template
from django.shortcuts import render
import pickle
import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def index(request): #esta función es una vista

    marcas = getMarcas()
    provincias = getProvinces()
    return render(request,'index.html',{"marcas":marcas,"provincias":provincias})


def prediction(request):
    marca = request.GET['marca']
    modelo = request.GET['modelo']
    anno = request.GET['anno']
    hp = request.GET['hp']
    km = request.GET['km']
    province = request.GET['province']
    cubicCapacity = request.GET['cC']
    gas = request.GET['gas']
    transmission = request.GET['change']
    doors = request.GET['doors']

    predictiomRandomForest = getRandomForest(int(gas),int(km),int(marca),int(modelo),int(province),int(transmission),int(anno),int(cubicCapacity),int(doors),int(hp))
    predictionkNN = getkNN(int(gas),int(km),int(marca),int(modelo),int(province),int(transmission),int(anno),int(cubicCapacity),int(doors),int(hp))
    predictionBayesian = getBayesianRidge(int(gas),int(km),int(marca),int(modelo),int(province),int(transmission),int(anno),int(cubicCapacity),int(doors),int(hp))
    predictionLinearRegressor = getLinearRegressor(int(gas),int(km),int(marca),int(modelo),int(province),int(transmission),int(anno),int(cubicCapacity),int(doors),int(hp))
    predictionTensorFlow = getTensorFlow(int(gas),int(km),int(marca),int(modelo),int(province),int(transmission),int(anno),int(cubicCapacity),int(doors),int(hp))

    return render(request,'prediction.html',{
        'predictionRandomForest':predictiomRandomForest,
        'predictionkNN':predictionkNN,
        'predictionBayesian':predictionBayesian,
        'predictionLinearRegressor':predictionLinearRegressor,
        'predicitionTensorFlow':predictionTensorFlow[0]
        })




def getMarcas():
    marcas = {
  'ALFA ROMEO': 1,
  'AUDI': 4,
  'BMW': 7,
  'CHEVROLET': 9,
  'CITROEN': 11,
  'CUPRA': 1400,
  'DACIA': 1011,
  'DS': 1358,
  'FIAT': 14,
  'FORD': 15,
  'HONDA': 69,
  'HYUNDAI': 18,
  'JAGUAR': 20,
  'JEEP': 21,
  'KIA': 22,
  'LAND-ROVER': 24,
  'LEXUS': 25,
  'MASERATI': 26,
  'MAZDA': 27,
  'MERCEDES-BENZ': 28,
  'MINI': 222,
  'MITSUBISHI': 30,
  'NISSAN': 31,
  'OPEL': 32,
  'PEUGEOT': 33,
  'PORSCHE': 34,
  'RENAULT': 35,
  'SEAT': 39,
  'SKODA': 40,
  'SMART': 41,
  'SSANGYONG': 42,
  'TESLA': 1354,
  'TOYOTA': 46,
  'VOLKSWAGEN': 47,
  'VOLVO': 48}
    return marcas


def getProvinces():
    provincias = {
  1: 'Alava',
  2: 'Albacete',
  3: 'Alicante',
  4: 'Almeria',
  5: 'Avila',
  6: 'Badajoz',
  7: 'Baleares',
  8: 'Barcelona',
  9: 'Burgos',
  10: 'Caceres',
  11: 'Cadiz',
  12: 'Castellon',
  13: 'Ciudad Real',
  14: 'Cordoba',
  15: 'A Coruna',
  16: 'Cuenca',
  17: 'Girona',
  18: 'Granada',
  19: 'Guadalajara',
  20: 'Guipuzcoa',
  21: 'Huelva',
  22: 'Huesca',
  23: 'Jaen',
  24: 'Leon',
  25: 'Lleida',
  26: 'La Rioja',
  27: 'Lugo',
  28: 'Madrid',
  29: 'Malaga',
  30: 'Murcia',
  31: 'Navarra',
  32: 'Orense',
  33: 'Asturias',
  34: 'Palencia',
  35: 'Las Palmas',
  36: 'Pontevedra',
  37: 'Salamanca',
  38: 'Sta. C. Tenerife',
  39: 'Cantabria',
  40: 'Segovia',
  41: 'Sevilla',
  43: 'Tarragona',
  44: 'Teruel',
  45: 'Toledo',
  46: 'Valencia',
  47: 'Valladolid',
  48: 'Vizcaya',
  49: 'Zamora',
  50: 'Zaragoza'
  }
    return provincias


# MÉTODOS PARA LA OBTENCIÓN DE LAS PREDICCIONES

def getRandomForest(gas,km,marca,modelo,province,transmission,anno,cubicCapacity,doors,hp):
    fileModelRandomForest = 'D:/Grado IA-Big Data/Programación_de_Inteligencia_artificial/Carmelo/Entorno-Django/Django-Predict-Model-Web/cochesNet/cochesNet/static/Models/pickle_model_RandomForest.pkl'
    with open(fileModelRandomForest, 'rb') as file:
        pickle_model = pickle.load(file)
    
    # Calculate the accuracy score and predict target values
    YpredictRandomFores = pickle_model.predict([[gas,km,marca,modelo,province,transmission,anno,marca,modelo,cubicCapacity,doors,hp]])
    return round(YpredictRandomFores[0],2)


def getkNN(gas,km,marca,modelo,province,transmission,anno,cubicCapacity,doors,hp):
    fileModel = 'D:/Grado IA-Big Data/Programación_de_Inteligencia_artificial/Carmelo/Entorno-Django/Django-Predict-Model-Web/cochesNet/cochesNet/static/Models/pickle_model_kNN.pkl'
    with open(fileModel, 'rb') as file:
        pickle_model = pickle.load(file)
    
    # Calculate the accuracy score and predict target values
    predict = pickle_model.predict(np.array([[gas,km,marca,modelo,province,transmission,anno,marca,modelo,cubicCapacity,doors,hp]]))
    return round(predict[0],2)


def getBayesianRidge(gas,km,marca,modelo,province,transmission,anno,cubicCapacity,doors,hp):
    fileModel = 'D:/Grado IA-Big Data/Programación_de_Inteligencia_artificial/Carmelo/Entorno-Django/Django-Predict-Model-Web/cochesNet/cochesNet/static/Models/pickle_model_bayesianRig.pkl'
    with open(fileModel, 'rb') as file:
        pickle_model = pickle.load(file)
    
    # Calculate the accuracy score and predict target values
    predict = pickle_model.predict(np.array([[gas,km,marca,modelo,province,transmission,anno,marca,modelo,cubicCapacity,doors,hp]]))
    return round(predict[0],2)

def getLinearRegressor(gas,km,marca,modelo,province,transmission,anno,cubicCapacity,doors,hp):
    fileModel = 'D:/Grado IA-Big Data/Programación_de_Inteligencia_artificial/Carmelo/Entorno-Django/Django-Predict-Model-Web/cochesNet/cochesNet/static/Models/pickle_model_LinearRegressor.pkl'
    with open(fileModel, 'rb') as file:
        pickle_model = pickle.load(file)
    
    # Calculate the accuracy score and predict target values
    predict = pickle_model.predict(np.array([[gas,km,marca,modelo,province,transmission,anno,marca,modelo,cubicCapacity,doors,hp]]))
    return round(predict[0],2)


def getTensorFlow(gas,km,marca,modelo,province,transmission,anno,cubicCapacity,doors,hp):
    from sklearn import preprocessing
    import tensorflow as tf

    dataPreScaling = [[gas,km,marca,modelo,province,transmission,anno,marca,modelo,cubicCapacity,doors,hp]]

    scaler = preprocessing.StandardScaler().fit(dataPreScaling)
    dataScaled = scaler.transform(dataPreScaling)

    fileModel = 'D:/Grado IA-Big Data/Programación_de_Inteligencia_artificial/Carmelo/Entorno-Django/Django-Predict-Model-Web/cochesNet/cochesNet/static/Models/TensorFlow3'
    new_model = tf.keras.models.load_model('D:/Grado IA-Big Data/Programación_de_Inteligencia_artificial/Carmelo/Entorno-Django/Django-Predict-Model-Web/cochesNet/cochesNet/static/Models/TensorFlow3')

    
    predict = new_model.predict(dataScaled)

    return scaler.inverse_transform(predict)
