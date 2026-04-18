import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2
from datetime import datetime

# Fetch variables
USER = "postgres.aldnzmuikliytxwuyjjy"
PASSWORD = "adrADR0924."
HOST = "aws-1-us-east-1.pooler.supabase.com"
PORT = "6543"
DBNAME = "postgres"

# Configuración de la página
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")

# Función para conectar a la base de datos
def get_connection():
    return psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME
    )

# Crear tabla si no existe
def init_db():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predicciones (
                id SERIAL PRIMARY KEY,
                sepal_length FLOAT,
                sepal_width FLOAT,
                petal_length FLOAT,
                petal_width FLOAT,
                especie_predicha VARCHAR(50),
                confianza FLOAT,
                fecha_creacion TIMESTAMP DEFAULT NOW()
            );
        """)
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"Error al inicializar la base de datos: {e}")

# Guardar predicción en la base de datos
def save_prediction(sepal_length, sepal_width, petal_length, petal_width, especie, confianza):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predicciones (sepal_length, sepal_width, petal_length, petal_width, especie_predicha, confianza)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (sepal_length, sepal_width, petal_length, petal_width, especie, confianza))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error al guardar: {e}")
        return False

# Obtener historial ordenado de forma DESCENDENTE por fecha_creacion
def get_history():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, sepal_length, sepal_width, petal_length, petal_width,
                   especie_predicha, confianza, fecha_creacion
            FROM predicciones
            ORDER BY fecha_creacion DESC  -- ← Orden descendente por fecha
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"Error al obtener historial: {e}")
        return []

# Inicializar DB
init_db()

# Función para cargar los modelos
@st.cache_resource
def load_models():
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo en la carpeta 'components/'")
        return None, None, None

# Título
st.title("🌸 Predictor de Especies de Iris")

# Cargar modelos
model, scaler, model_info = load_models()

if model is not None:
    st.header("Ingresa las características de la flor:")

    sepal_length = st.number_input("Longitud del Sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width  = st.number_input("Ancho del Sépalo (cm)",    min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    petal_length = st.number_input("Longitud del Pétalo (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    petal_width  = st.number_input("Ancho del Pétalo (cm)",    min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    if st.button("Predecir Especie"):
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        features_scaled = scaler.transform(features)

        prediction    = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        target_names      = model_info['target_names']
        predicted_species = target_names[prediction]
        confidence        = max(probabilities)

        st.success(f"Especie predicha: **{predicted_species}**")
        st.write(f"Confianza: **{confidence:.1%}**")

        st.write("Probabilidades:")
        for species, prob in zip(target_names, probabilities):
            st.write(f"- {species}: {prob:.1%}")

        # Guardar en base de datos
        saved = save_prediction(sepal_length, sepal_width, petal_length, petal_width,
                                predicted_species, float(confidence))
        if saved:
            st.info("✅ Predicción guardada en la base de datos.")

    # Historial de predicciones (descendente por fecha)
    st.divider()
    st.header("📋 Historial de Predicciones")
    st.caption("Ordenado de más reciente a más antiguo")

    if st.button("🔄 Actualizar historial"):
        st.rerun()

    history = get_history()

    if history:
        for row in history:
            id_, s_len, s_wid, p_len, p_wid, especie, confianza, fecha = row
            with st.expander(f"#{id_} — {especie}  |  {fecha.strftime('%Y-%m-%d %H:%M:%S')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"🌿 **Sépalo:** {s_len} x {s_wid} cm")
                    st.write(f"🌸 **Pétalo:** {p_len} x {p_wid} cm")
                with col2:
                    st.write(f"🏷️ **Especie:** {especie}")
                    st.write(f"📊 **Confianza:** {confianza:.1%}")
    else:
        st.write("No hay predicciones registradas aún.")
