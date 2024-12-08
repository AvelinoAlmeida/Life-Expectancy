import streamlit as st
import h2o
import pandas as pd
import numpy as np

# Inicializar o H2O
h2o.init()

# Carregar os modelos treinados
try:
    socioeconomic_model_path = "./models/rf_socioeconomic_model"
    vaccination_model_path = "./models/rf_vaccination_model"

    socioeconomic_model = h2o.load_model(socioeconomic_model_path)
    vaccination_model = h2o.load_model(vaccination_model_path)

    st.sidebar.success("Modelos carregados com sucesso!")
except Exception as e:
    st.sidebar.error(f"Erro ao carregar os modelos: {e}")
    st.stop()

# Página inicial
st.title("🌍 Sistema de Apoio à Decisão - Expectativa de Vida")
st.sidebar.title("Menu de Navegação")
menu = st.sidebar.radio("Ir para:", ["Início", "Socioeconômico", "Vacinação"])

# Dados de entrada
def get_feature_ranges(feature_list, data_path):
    """
    Obtém os valores mínimos e máximos para cada variável no dataset.
    """
    df = pd.read_csv(data_path)
    return df[feature_list].min().astype(float), df[feature_list].max().astype(float)

# Página inicial
if menu == "Início":
    st.markdown("""
    <h2>🌍 Sistema de Apoio à Decisão - Expectativa de Vida</h2>
    <p>Este sistema utiliza modelos preditivos para ajudar a estimar a expectativa de vida com base em variáveis socioeconômicas e taxas de vacinação.</p>
    """, unsafe_allow_html=True)

# Página para o modelo Socioeconômico
elif menu == "Socioeconômico":
    st.header("📈 Previsão de Expectativa de Vida - Modelo Socioeconômico")

    # Variáveis do modelo socioeconômico
    socioeconomic_features = ['percentage expenditure', 'Total expenditure', 'GDP', 'Income composition of resources', 'Schooling']
    min_values, max_values = get_feature_ranges(socioeconomic_features, './data/Life_Expectancy_Clean.csv')

    # Capturar entrada do usuário
    input_data = {}
    for i, feature in enumerate(socioeconomic_features):
        input_data[feature] = st.slider(f"{feature}", min_values[i], max_values[i], (min_values[i] + max_values[i]) / 2)

    # Fazer a previsão
    if st.button("Fazer Previsão - Modelo Socioeconômico"):
        input_df = pd.DataFrame([input_data])
        input_h2o = h2o.H2OFrame(input_df)

        prediction = socioeconomic_model.predict(input_h2o).as_data_frame().iloc[0, 0]
        h2o.remove(input_h2o)  # Limpar o H2OFrame após a previsão

        st.write(f"**Expectativa de Vida Prevista (Socioeconômico):** {round(prediction, 2)} anos")

# Página para o modelo de Vacinação
elif menu == "Vacinação":
    st.header("📈 Previsão de Expectativa de Vida - Modelo de Vacinação")

    # Variáveis do modelo de vacinação
    vaccination_features = ['Hepatitis B', 'Polio', 'Diphtheria', 'percentage expenditure']
    min_values, max_values = get_feature_ranges(vaccination_features, './data/Life_Expectancy_Clean.csv')

    # Capturar entrada do usuário
    input_data = {}
    for i, feature in enumerate(vaccination_features):
        input_data[feature] = st.slider(f"{feature}", min_values[i], max_values[i], (min_values[i] + max_values[i]) / 2)

    # Fazer a previsão
    if st.button("Fazer Previsão - Modelo de Vacinação"):
        input_df = pd.DataFrame([input_data])
        input_h2o = h2o.H2OFrame(input_df)

        prediction = vaccination_model.predict(input_h2o).as_data_frame().iloc[0, 0]
        h2o.remove(input_h2o)  # Limpar o H2OFrame após a previsão

        st.write(f"**Expectativa de Vida Prevista (Vacinação):** {round(prediction, 2)} anos")
