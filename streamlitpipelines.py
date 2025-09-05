import streamlit as st
import pandas as pd
from joblib import load
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
#import pyautogui

# Cargar el modelo de regresión
regressor = load('Modelopipeline.joblib')

# Cargar el encoder
#with open('encoderpipeline.pickle', 'rb') as f:
#    encoder = pickle.load(f)

# Inicializar variables
rd_spend = administration = marketing_spend = 0.0
selected_state = "New York"

# Streamlit app
st.title("Modelo de Regresión")
st.markdown("##### Si colocas un valor negativo, aparecerá un error y no podrás completar otros campos. La predicción será incorrecta.")

# Sidebar para la entrada del usuario
st.sidebar.header("Campos a Evaluar")

# Entrada del usuario para RD_Spend
rd_spend = st.sidebar.number_input("**RD_Spend (Min=0, Max=165349.20)**", min_value=0.0, value=float(rd_spend))

# Entrada del usuario para Administration
administration = st.sidebar.number_input("**Administration (Min=51283.14, Max=182645.56)**", min_value=0.0, value=float(administration))

# Entrada del usuario para Marketing-Spend
marketing_spend = st.sidebar.number_input("**Marketing Spend (Min=0, Max=471784.10)**", min_value=0.0, value=float(marketing_spend))

# Entrada del usuario para State
st.sidebar.markdown("<h1 style='font-size: 24px;'>State</h1>", unsafe_allow_html=True)
selected_state = st.sidebar.selectbox("Select State", ["New York", "California", "Florida"], index=["New York", "California", "Florida"].index(selected_state))
# Función para resetear las entradas
def reset_inputs():
    global rd_spend, administration, marketing_spend, selected_state
    rd_spend = administration = marketing_spend = 0.0
    selected_state = "New York"

# Botón para predecir
if st.sidebar.button("Predecir"):
    # Validar las entradas
    if all(isinstance(val, (int, float)) and val >= 0 for val in [rd_spend, administration, marketing_spend]):
        # Crear un DataFrame con las entradas del usuario
        obs = pd.DataFrame({
            'R&D Spend': [rd_spend],
            'Administration': [administration],
            'Marketing Spend': [marketing_spend],
            'State': [selected_state]
        })

        # Mostrar el DataFrame de entradas para depuración
        st.write("DataFrame de Entradas:")
        st.write(obs)

        #-----------------------Todo esto lo hace el pipeline-------------------------
        # Escalar las columnas numéricas----------------------------------------------
        #numeric_cols = obs.select_dtypes(include=np.number).columns.tolist()
        #scaler = StandardScaler()
        # means es el vector de medias de la base de datos original, para cuantitativas
        #means = pd.DataFrame([73721.6156, 121344.6396, 211025.0978])
        #std_devs = pd.DataFrame([45902.256482,  28017.802755, 122290.310726])

        # Utilizar .values.flatten() para obtener un array unidimensional
        #scaler.mean_ = means.values.flatten()
        #scaler.scale_ = std_devs.values.flatten()
        #obs[numeric_cols] = scaler.transform(obs[numeric_cols])
        #---------------------------------------------------------------------------------
        # Codificar las columnas categóricas directamente usando el encoder
        #categorical_cols = obs.select_dtypes('object').columns.tolist()
        #encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
        #obs_encoded = pd.DataFrame(encoder.transform(obs[categorical_cols]), columns=encoded_cols)

        # Combinar columnas numéricas y codificadas categóricas
        #obs = pd.concat([obs[numeric_cols], obs_encoded], axis=1)

        # Predecir usando el modelo
        target = regressor.predict(obs)

        # Mostrar la predicción con un tamaño de fuente grande usando markdown
        st.markdown(f'<p style="font-size: 40px; color: green;">La predicción del Profit será: ${target[0]:,.2f}</p>', unsafe_allow_html=True)

    else:
        st.warning("Rellene todos los espacios en blanco")

# Colocar el botón "Resetear" debajo del botón "Predecir"
if st.sidebar.button("Resetear"):
    # Resetear inputs
    reset_inputs()
    # Recargar la página usando pyautogui
    #pyautogui.hotkey("ctrl", "f5")
    
#   R&D Spend	Administration	Marketing Spend  Ciudad  
#	  142107.34  	91391.77	366168.42         Florida    ---->

# Cambiar los valores.
# Para asignar valores: ver los rangos de las cuantitativas ( MÍNIMO --MÁXIMO)
# eso determinan  cómo predice el modelo. 

#  streamlit run streamlitpipelines.py       en la consola
#  pip freeze > requirements.txt