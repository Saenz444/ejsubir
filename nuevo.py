# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.inspection import permutation_importance
import multiprocessing

# Leer el conjunto de datos con pandas
archivo_excel = 'Data.xlsx'
Data = pd.read_excel(archivo_excel)

# Eliminar las columnas 'estudiante' y 'si_no'
Data = Data.drop(['estudiante', 'si_no'], axis=1)

# Crear la aplicación Streamlit
def main():
    st.title('Aplicación Streamlit para Modelo de Clasificación')

    # Ajuste del modelo RandomForest
    X_train, X_test, y_train, y_test = train_test_split(
        Data.drop(columns='ingreso'),
        Data['ingreso'],
        random_state=123
    )
    model_rf = fit_random_forest(X_train, y_train)

    # Mostrar la matriz de confusión
    st.subheader('Matriz de Confusión - RandomForest:')
    plot_confusion_matrix(model_rf, X_test, y_test)

    # Mostrar el gráfico de importancia de características para RandomForest
    st.subheader('Gráfico de Importancia de Características para RandomForest:')
    plot_feature_importance_rf(X_train, y_train)

# Función para ajustar el modelo RandomForest
def fit_random_forest(X_train, y_train):
    param_grid = {'n_estimators': [150],
                  'max_features': [3, 5, 7],
                  'max_depth': [None, 3, 10, 20],
                  'criterion': ['gini', 'entropy']}

    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=123),
        param_grid=param_grid,
        scoring='accuracy',
        n_jobs=multiprocessing.cpu_count() - 1,
        cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=123),
        refit=True,
        verbose=0,
        return_train_score=True
    )
    grid.fit(X=X_train, y=y_train)
    
    return grid.best_estimator_

# Función para mostrar la matriz de confusión
def plot_confusion_matrix(model, X_test, y_test):
    predicciones = model.predict(X=X_test)
    mat_confusion = confusion_matrix(y_true=y_test, y_pred=predicciones)
    accuracy_rf = accuracy_score(y_true=y_test, y_pred=predicciones)
    st.write(f'{accuracy_rf:.2%}')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(mat_confusion, annot=True, fmt='d', cmap='YlGnBu', cbar=False)
    plt.xlabel('Etiqueta Real')
    plt.ylabel('Etiqueta Predicha')
    plt.title('Matriz de Confusión - RandomForest')
    st.pyplot()
    
    

# Función para mostrar el gráfico de importancia de características para RandomForest
def plot_feature_importance_rf(X_train, y_train):
    model_rf = RandomForestClassifier(random_state=123)
    model_rf.fit(X_train, y_train)

    importancia = permutation_importance(
        estimator=model_rf,
        X=X_train,
        y=y_train,
        n_repeats=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=multiprocessing.cpu_count() - 1,
        random_state=123
    )

    df_importancia = pd.DataFrame(
        {k: importancia[k] for k in ['importances_mean', 'importances_std']}
    )
    df_importancia['predictor'] = X_train.columns

    color = ['r', 'r', 'r', 'y', 'g', 'g', 'g']
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.barh(
        df_importancia['predictor'],
        df_importancia['importances_mean'],
        xerr=df_importancia['importances_std'],
        align='center',
        alpha=1,
        color=color
    )
    ax.plot(
        df_importancia['importances_mean'],
        df_importancia['predictor'],
        marker="o",
        linestyle="",
        alpha=0.8,
        color="r"
    )
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=8)
    plt.grid(alpha=.5)
    st.pyplot()
###############################################
#grafico de coeficientes rl sacado del web final
# Gráfico de barras para mostrar coeficientes de la regresión
        plt.figure(figsize=(10, 6))
        coeficientes = [interceptor_RL] + list(coeficientes_RL)
        variables = ['V-0'] + [f'P-{i}' for i in range(1, 8)]
        sns.barplot(x=coeficientes, y=variables)
        plt.xlabel('Coeficientes')
        plt.ylabel('Variables')
        plt.title('Coeficientes de la Regresión Lineal')
        st.pyplot(plt)
        plt.clf()

if __name__ == '__main__':
    main()
