# Archivo: alternativa3.py
# tercera alternativa

# Cargar paquetes
import numpy as np      
import pandas as pd     
import matplotlib.pyplot as plt 

# Modulos para redes neuronales
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras import regularizers
from keras.regularizers import l2
from keras.optimizers.schedules import ExponentialDecay

# Tratamiento de datos
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#######################################
## Carga y preprocesamiento de datos ## 
#######################################

# Cargamos los datos desde la URL pública
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip', compression='zip', header=0, na_values=' ?')

# Establecemos los nombres de columnas
data = data.set_axis(['age','job','marital','education','default','balance','housing',
                      'loan','contact','day','month','duration','campaign','pdays','previous',
                     'poutcome','y'], axis=1)

# Mostramos las primeras filas
data.head(10)

# Identificamos valores perdidos
data.isna().sum()

# Tipo de datos del dataframe
data.dtypes

# Validación de variables categóricas
print(data['job'].unique())
print(data['marital'].unique())
print(data['education'].unique())
print(data['default'].unique())
print(data['housing'].unique())
print(data['loan'].unique())
print(data['contact'].unique())
print(data['month'].unique())
print(data['poutcome'].unique())

# Manejo de desbalanceo con oversampling
data.shape
filtro = data['y'] == 'yes'
print("Originales: ", sum(filtro), "\n")
nuevos = data[filtro].sample(n=3479, replace=True)
data_ampliado = pd.concat([data, nuevos])
data_ampliado.shape

# Separamos características y variable objetivo
X = data_ampliado.iloc[:, :-1]
y = data_ampliado.iloc[:, -1]

# Transformamos la etiqueta en 0 y 1
y = pd.Categorical(y).codes

# División en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    shuffle=True, random_state=123,
                                                    stratify=y)

# Nombres de variables numéricas y categóricas
num_var = X.select_dtypes(include=['number']).columns.values
cat_var = X.select_dtypes(include=['object']).columns.values

# Preprocesamiento con transformer
transformer = make_column_transformer(
    (StandardScaler(), num_var),
    (OneHotEncoder(), cat_var),
    verbose_feature_names_out=False)

# Ajustamos el transformer con datos de entrenamiento
transformer.fit(x_train)

# Transformamos datos
x_train = transformer.transform(x_train)
x_train = pd.DataFrame(x_train, columns=transformer.get_feature_names_out())
x_test = transformer.transform(x_test)
x_test = pd.DataFrame(x_test, columns=transformer.get_feature_names_out())

# Crear modelo de red neuronal
modelo3 = Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(16, activation="relu", kernel_initializer="he_normal"),
    Dropout(0.1),
    Dense(8, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    Dense(1, activation="sigmoid")
])

# Resumen del modelo
modelo3.summary()

# Ajustar tasa de aprendizaje
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=500,
    decay_rate=0.9)

# Compilar el modelo
modelo3.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Entrenamiento con early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history3 = modelo3.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)

# Visualización de resultados
print(history3.history.keys())
df3 = pd.DataFrame(history3.history)
df3.plot()

# Gráficos de precisión y pérdida
dfAccuracy3 = df3.loc[:,["accuracy","val_accuracy"]]
dfAccuracy3.plot()
plt.savefig('images/accuracy_plot.png')

dfLoss3 = df3.loc[:,["loss","val_loss"]]
dfLoss3.plot()
plt.savefig('images/loss_plot.png')

# Evaluación en conjunto de prueba
metrics3 = modelo3.evaluate(x_test, y_test)
print(f'Tercera Alternativa - Pérdida: {metrics3[0]}, Precisión: {metrics3[1]}')

# Predicciones
predicciones3 = modelo3.predict(x_test)
predic_test3 = 1*(predicciones3>0.5)

# Matriz de confusión
mc3 = confusion_matrix(y_test, predic_test3)
print(mc3)

# Gráfico de matriz de confusión
class_names = ['NoRealiza', 'Realiza']
disp = ConfusionMatrixDisplay(confusion_matrix=mc3, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, values_format='g')
plt.savefig('images/confusion_matrix.png')

# Guardar el modelo
modelo3.save('alternativa3.h5')
