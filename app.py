# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:12:22 2026

@author: HECTOR
"""

import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Configuración de la interfaz
st.set_page_config(page_title="Dashboard de Análisis y Predicción", layout="wide")
st.title("Plataforma de Análisis de Datos y Predicción de Demanda")

# 2. Carga de archivo CSV y conversión a SQL
st.sidebar.header("1. Cargar Datos")
archivo_csv = st.sidebar.file_uploader("Sube tu archivo CSV", type=["csv"])

if archivo_csv is not None:
    # Leer CSV
    df = pd.read_csv(archivo_csv)
    
    # Crear conexión SQLite en memoria (o en un archivo local .db si prefieres persistencia)
    conn = sqlite3.connect('base_datos_local.db')
    # Convertir DataFrame a tabla SQL
    df.to_sql('datos_cargados', conn, if_exists='replace', index=False)
    
    st.sidebar.success("✅ Datos cargados y convertidos a SQL exitosamente.")
    
    # Mostrar pestañas para organizar la interfaz
    tab1, tab2, tab3 = st.tabs(["📊 Análisis Estadístico", "🤖 Consultas NLP", "📈 Predicción de Demanda"])

    # --- PESTAÑA 1: ANÁLISIS ESTADÍSTICO ---
    with tab1:
        st.header("Análisis Estadístico Descriptivo")
        st.dataframe(df.head()) # Vista previa
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Estadísticas Generales")
            st.write(df.describe())
        
        with col2:
            st.subheader("Distribución de Datos")
            columna_plot = st.selectbox("Selecciona una columna para ver su distribución:", df.select_dtypes(include=np.number).columns)
            fig, ax = plt.subplots()
            ax.hist(df[columna_plot].dropna(), bins=20, color='skyblue', edgecolor='black')
            st.pyplot(fig)

    # --- PESTAÑA 2: CONSULTAS EN LENGUAJE NATURAL ---
    with tab2:
        st.header("Consultas a la Base de Datos")
        st.info("Para procesar lenguaje natural real a SQL, se requiere integrar la API de un LLM (Ej. Gemini, OpenAI) o usar la librería `pandasai`.")
        
        consulta_texto = st.text_input("Pregunta algo sobre tus datos (Ej. 'Mostrar ventas mayores a 100'):")
        
        if st.button("Ejecutar Consulta"):
            # AQUI VA LA LÓGICA DEL LLM. 
            # Como ejemplo básico sin API, simularemos una búsqueda de texto simple en el dataframe:
            try:
                # Si estuvieras usando una API de LLM, el modelo traduciría "consulta_texto" a esta query SQL:
                # query_sql = llm.translate_to_sql(consulta_texto)
                # resultado = pd.read_sql_query(query_sql, conn)
                
                st.write(f"Has consultado: *{consulta_texto}*")
                st.warning("Implementa aquí tu API Key para habilitar la traducción de NLP a SQL.")
            except Exception as e:
                st.error(f"Error en la consulta: {e}")

    # --- PESTAÑA 3: PREDICCIÓN DE DEMANDA ---
    with tab3:
        st.header("Modelo de Predicción de Demanda")
        st.write("Selecciona tus variables para proyectar el comportamiento futuro.")
        
        # Seleccionar variables
        columnas_numericas = df.select_dtypes(include=np.number).columns.tolist()
        
        if len(columnas_numericas) >= 2:
            col_x = st.selectbox("Variable de Tiempo / Índice (X):", columnas_numericas)
            col_y = st.selectbox("Variable a Predecir / Demanda (Y):", columnas_numericas, index=1)
            
            dias_futuros = st.slider("¿Cuántos periodos hacia el futuro deseas predecir?", 1, 30, 7)
            
            if st.button("Generar Predicción"):
                # Preparar datos
                X = df[[col_x]].values
                y = df[col_y].values
                
                # Entrenar modelo
                # Aunque construir el algoritmo de descenso de gradiente desde cero con NumPy es excelente 
                # para dominar las matemáticas detrás del modelo, en interfaces de producción como esta, 
                # instanciar la clase LinearRegression de scikit-learn es mucho más eficiente y estable.
                modelo = LinearRegression()
                modelo.fit(X, y)
                
                # Crear datos futuros
                ultimo_valor_x = X[-1][0]
                paso_x = (X[-1][0] - X[0][0]) / len(X) if len(X) > 1 else 1
                X_futuro = np.array([[ultimo_valor_x + i * paso_x] for i in range(1, dias_futuros + 1)])
                
                # Predecir
                predicciones = modelo.predict(X_futuro)
                
                # Graficar resultados
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                ax2.scatter(X, y, color='blue', label='Datos Históricos')
                ax2.plot(X, modelo.predict(X), color='black', linestyle='--', label='Ajuste Lineal')
                ax2.scatter(X_futuro, predicciones, color='red', label='Predicción Futura')
                ax2.set_xlabel(col_x)
                ax2.set_ylabel(col_y)
                ax2.legend()
                ax2.grid(True)
                
                st.pyplot(fig2)
                
                # Mostrar tabla de predicciones
                df_pred = pd.DataFrame({f"Futuro {col_x}": X_futuro.flatten(), "Demanda Proyectada": predicciones})
                st.dataframe(df_pred)
        else:
            st.warning("El CSV necesita al menos dos columnas numéricas para realizar una regresión.")
else:
    st.info("👈 Por favor, sube un archivo CSV en la barra lateral para comenzar.")
