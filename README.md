# Predicción de Suscripciones Bancarias
Modelo predictivo para predecir suscripciones a depósitos bancarios usando machine learning.

## Introducción
Este proyecto utiliza una red neuronal para predecir si un cliente suscribirá un depósito a plazo basado en datos de campañas de marketing telefónico de un banco portugués. El objetivo es optimizar estrategias de marketing, reduciendo costos operativos y mejorando tasas de conversión, un caso de uso relevante para el sector financiero europeo. Se aborda el desbalanceo de datos y se implementa un modelo robusto con técnicas avanzadas de machine learning.

## Dataset

* **Fuente** : UCI Machine Learning Repository (Bank Marketing Dataset).
* **Descripción**: Contiene 45,211 registros de campañas de marketing telefónico con 16 características, incluyendo edad, empleo, estado civil, duración de llamada, entre otros. La variable objetivo (y) indica si el cliente suscribió un depósito a plazo (yes/no).

**Preprocesamiento** :
* Manejo de desbalanceo mediante oversampling (muestreo con reemplazo de la clase minoritaria).
* Normalización de variables numéricas y codificación one-hot de variables categóricas.



## Herramientas y Tecnologías

* **Python** : Pandas, NumPy, Scikit-learn, TensorFlow, Keras.
* **Visualización** : Matplotlib.
* **Entorno** : Google Colab para ejecución interactiva.

## Pasos del Proyecto

1. **Carga y Exploración** : Importación del dataset público desde UCI y análisis de valores perdidos.
2. **Preprocesamiento** : Balanceo de clases con oversampling, normalización (StandardScaler) y codificación one-hot (OneHotEncoder).
3. **Modelado** : Red neuronal con capas densas, regularización L2, dropout y optimizador Adam con decaimiento de tasa de aprendizaje.
4. **Entrenamiento** : Uso de early stopping para evitar sobreajuste, con 50 épocas y validación cruzada.
5. **Evaluación** : Cálculo de precisión, pérdida y matriz de confusión en el conjunto de prueba.

## Resultados

**Precisión** : 92.44% en el conjunto de prueba.
**Pérdida** : 0.2189.
**Matriz de Confusión**:

* Verdaderos Positivos: 781 (clientes que suscriben correctamente predichos).
* Falsos Positivos: 102.
* Verdaderos Negativos: 698.
* Falsos Negativos: 19.


* **Insights**: El modelo identifica con alta precisión clientes propensos a suscribir, permitiendo enfocar esfuerzos de marketing en segmentos de alto valor, lo que puede reducir costos hasta un 20% en campañas similares.


## Ejecución Interactiva
Ver en Google Colab para explorar el código y resultados en tiempo real.

## Conclusión
Este proyecto demuestra mi capacidad para combinar técnicas de machine learning con un enfoque orientado al negocio, abordando desafíos como el desbalanceo de datos y optimizando estrategias de marketing. Mi experiencia en finanzas (SAP FI) y transición a Data Analytics me permite generar soluciones que cumplen con normativas europeas como RGPD, aportando valor al sector financiero.
## Contacto

- **LinkedIn**: linkedin.com/in/ingridortizmoreno
- **Email** : jobiso88@gmail.com
