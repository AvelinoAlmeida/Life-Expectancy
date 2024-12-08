import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configurar o Streamlit
st.set_page_config(page_title="Previsão de Expectativa de Vida", layout="wide")

# Barra lateral de navegação
st.sidebar.title("Menu de Navegação")
menu = st.sidebar.radio("Ir para", ["Início", "Estatísticas"    , "Socioeconômico", "Vacinação" ])

# Carregar os modelos salvos
try:
    socioeconomic_model = joblib.load("./models/socioeconomic_predictions.pkl")
    vaccination_model = joblib.load("./models/vaccination_predictions.pkl")
    st.sidebar.success("Modelos carregados com sucesso!")
except FileNotFoundError:
    st.sidebar.error("Erro: Um ou mais modelos não foram encontrados. Verifique o diretório './models'.")
    st.stop()

# Página inicial (Início)
if menu == "Início":
    st.markdown("<h2>🌍 Previsão de Expectativa de Vida</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p>Este sistema ajuda a prever a expectativa de vida com base em variáveis demográficas, clínicas e socioeconômicas.</p>",
        unsafe_allow_html=True
    )

# Página de análises
elif menu == "Estatísticas":
    st.title("📊 Estatísticas e Análises dos Dados")

    # Carregar um dataset para análises
    try:
        df = pd.read_csv("./data/Life_Expectancy_Clean.csv")
        st.success("Dataset carregado com sucesso!")
        
        # Estatísticas gerais
        st.subheader("Estatísticas Descritivas")
        st.write(df.describe())
        
        # Criar um gráfico interativo de correlação
        st.subheader("📊 Matriz de Correlação")
        corr_matrix = df.corr()
        st.write(corr_matrix.style.background_gradient(cmap="coolwarm"))

    except FileNotFoundError:
        st.error("Erro: O dataset './data/Life_Expectancy_Data.csv' não foi encontrado.")




# Página de previsão para o modelo Socioeconômico
elif menu == "Socioeconômico":
    st.title("📈 Previsão de Expectativa de Vida - Modelo Socioeconômico")
    st.write("Insira os dados abaixo para prever a expectativa de vida com base em fatores socioeconômicos:")

    # Variáveis do modelo socioeconômico (usadas no treinamento)
    feature_columns = ['percentage expenditure', 'Total expenditure', 'GDP', 'Income composition of resources', 'Schooling']

    # Criar uma entrada padrão para os dados de entrada
    input_data = {feature: st.slider(f"{feature}", 0.0, 100.0, 50.0) for feature in feature_columns}

    # Criar DataFrame para entrada do modelo
    input_df = pd.DataFrame([input_data])

    # Adicionar colunas ausentes com valores padrão (0.0) para garantir compatibilidade
    for column in socioeconomic_model.feature_names_in_:
        if column not in input_df.columns:
            input_df[column] = 0.0

    # Garantir a mesma ordem de colunas que foi usada no treinamento
    input_df = input_df[socioeconomic_model.feature_names_in_]

    # Fazer previsão
    if st.button("Fazer Previsão - Modelo Socioeconômico"):
        try:
            prediction = socioeconomic_model.predict(input_df)[0]
            st.write(f"**Expectativa de Vida Prevista (Socioeconômico):** {round(prediction, 2)} anos")
        except ValueError as e:
            st.error(f"Erro ao fazer a previsão: {e}")

    # Exibir a importância das variáveis
    st.subheader("📊 Importância das Variáveis no Modelo Socioeconômico")
    try:
        feature_importances = pd.DataFrame({
            "Variável": socioeconomic_model.feature_names_in_,
            "Importância": socioeconomic_model.feature_importances_
        }).sort_values(by="Importância", ascending=False)
        st.bar_chart(feature_importances.set_index("Variável"))
    except AttributeError:
        st.warning("A importância das variáveis não está disponível para o modelo carregado.")







# Página de previsão para o modelo de Vacinação
elif menu == "Vacinação":
    st.title("📈 Previsão de Expectativa de Vida - Modelo de Vacinação")
    st.write("Insira os dados abaixo para prever a expectativa de vida com base em taxas de vacinação:")

    # Variáveis do modelo de vacinação
    feature_columns = ['Hepatitis B', 'Polio', 'Diphtheria', 'percentage expenditure']

    # Criar uma entrada padrão para os dados de entrada
    input_data = {feature: st.slider(f"{feature}", 0.0, 100.0, 50.0) for feature in feature_columns}

    # Criar DataFrame para entrada do modelo
    input_df = pd.DataFrame([input_data])

    # Adicionar colunas ausentes com valores padrão (0.0) para garantir compatibilidade
    for column in vaccination_model.feature_names_in_:
        if column not in input_df.columns:
            input_df[column] = 0.0

    # Garantir a mesma ordem de colunas que foi usada no treinamento
    input_df = input_df[vaccination_model.feature_names_in_]

    # Fazer previsão
    if st.button("Fazer Previsão - Modelo de Vacinação"):
        try:
            prediction = vaccination_model.predict(input_df)[0]
            st.write(f"**Expectativa de Vida Prevista (Vacinação):** {round(prediction, 2)} anos")
        except ValueError as e:
            st.error(f"Erro ao fazer a previsão: {e}")
