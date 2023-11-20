# Importa las bibliotecas necesarias
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, jaccard_score, mean_squared_error

# Función para cargar datos
@st.cache_data
def cargar_datos():
    archivo_excel = 'Data.xlsx'
    data = pd.read_excel(archivo_excel)
    data = data.drop(columns=['estudiante', 'si_no'])
    return data

# Cargar los datos
data = cargar_datos()

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns='ingreso'),
    data['ingreso'],
    random_state=123
)

# Entrenar el modelo RandomForest
parametros_rf = {'n_estimators': 150, 'max_features': 5, 'max_depth': None, 'criterion': 'gini'}
modelo_rf = RandomForestClassifier(random_state=123, **parametros_rf)
modelo_rf.fit(X_train, y_train)

# Entrenar el modelo SVM
modelo_svm = SVC(kernel='rbf', C=1)
modelo_svm.fit(X_train, y_train)

# Entrenar el modelo KNN
modelo_knn = KNeighborsClassifier(n_neighbors=3)
modelo_knn.fit(X_train, y_train)

# Entrenar el modelo de Regresión Lineal
modelo_rl = LinearRegression()
modelo_rl.fit(X_train, y_train)

# Sidebar con hiperparámetros
st.sidebar.title('Hiperparámetros Utilizados (Random Forest)')
st.sidebar.write(f'n_estimators: {parametros_rf["n_estimators"]}')
st.sidebar.write(f'max_features: {parametros_rf["max_features"]}')
st.sidebar.write(f'max_depth: {parametros_rf["max_depth"]}')
st.sidebar.write(f'criterion: {parametros_rf["criterion"]}')

# Predicciones y evaluaciones RandomForest
predicciones_rf = modelo_rf.predict(X_test)
mat_confusion_rf = confusion_matrix(y_true=y_test, y_pred=predicciones_rf)
accuracy_rf = accuracy_score(y_true=y_test, y_pred=predicciones_rf)

# Matriz de confusión RandomForest
st.header('Matriz de Confusión - RandomForest')
plt.figure(figsize=(8, 6))
sns.heatmap(mat_confusion_rf, annot=True, fmt='d', cmap='YlGnBu', cbar=False)
plt.xlabel('Etiqueta Real')
plt.ylabel('Etiqueta Predicha')
plt.title('Matriz de Confusión - RandomForest')
st.pyplot()

# Precisión RandomForest
st.write(f'Precisión del modelo RandomForest: {accuracy_rf:.2f}')

# Predicciones y evaluaciones SVM
predicciones_svm = modelo_svm.predict(X_test)
mat_confusion_svm = confusion_matrix(y_true=y_test, y_pred=predicciones_svm)
accuracy_svm = accuracy_score(y_true=y_test, y_pred=predicciones_svm)

# Matriz de confusión SVM
st.header('Matriz de Confusión - SVM')
plt.figure(figsize=(8, 6))
sns.heatmap(mat_confusion_svm, annot=True, fmt='d', cmap='YlGnBu', cbar=False)
plt.xlabel('Etiqueta Real')
plt.ylabel('Etiqueta Predicha')
plt.title('Matriz de Confusión - SVM')
st.pyplot()

# Precisión SVM
st.write(f'Precisión del modelo SVM: {accuracy_svm:.2f}')

# Predicciones y evaluaciones KNN
predicciones_knn = modelo_knn.predict(X_test)
mat_confusion_knn = confusion_matrix(y_true=y_test, y_pred=predicciones_knn)
accuracy_knn = accuracy_score(y_true=y_test, y_pred=predicciones_knn)

# Matriz de confusión KNN
st.header('Matriz de Confusión - KNN')
plt.figure(figsize=(8, 6))
sns.heatmap(mat_confusion_knn, annot=True, fmt='d', cmap='YlGnBu', cbar=False)
plt.xlabel('Etiqueta Real')
plt.ylabel('Etiqueta Predicha')
plt.title('Matriz de Confusión - KNN')
st.pyplot()

# Precisión KNN
st.write(f'Precisión del modelo KNN: {accuracy_knn:.2f}')


# Predicciones y evaluaciones Regresión Lineal
predicciones_rl = modelo_rl.predict(X_test)
mse_rl = mean_squared_error(y_true=y_test, y_pred=predicciones_rl)
# Graficar las predicciones frente a los valores reales para Regresión Lineal
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predicciones_rl, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red', linewidth=2)
plt.xlabel('Valor Real')
plt.ylabel('Predicción')
plt.title('Regresión Lineal - Valores Reales vs Predicciones')
st.pyplot()

# ...

# Error cuadrático medio (MSE) Regresión Lineal
st.write(f'Error Cuadrático Medio (MSE) - Regresión Lineal: {mse_rl:.2f}')
