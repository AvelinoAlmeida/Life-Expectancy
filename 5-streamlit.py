import streamlit as st
import pandas as pd
import numpy as np
import h2o
from h2o.estimators import H2ORandomForestEstimator
from h2o.frame import H2OFrame
import plotly.express as px

# Inicializar o H2O
h2o.init()

# Carregar o modelo salvo
model_path = "./models/rf_vaccination_model"
rf_model = h2o.load_model(model_path)

# Configurar o Streamlit
st.set_page_config(page_title="Previsão de Expectativa de Vida", layout="wide")

# Barra lateral de navegação
st.sidebar.title("Menu de Navegação")
menu = st.sidebar.radio("Ir para", ["Início", "Estatísticas", "Previsão", "Previsão em Massa"])

# Página inicial (Início)
if menu == "Início":
    st.markdown("<h2>🌍 Previsão de Expectativa de Vida</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p>Este sistema utiliza um modelo de Random Forest para prever a expectativa de vida com base em taxas de vacinação.</p>",
        unsafe_allow_html=True
    )

# Página de análises estatísticas
elif menu == "Estatísticas":
    st.title("📊 Estatísticas e Análises dos Dados")

    # Carregar o dataset original
    try:
        df = pd.read_csv("./data/Life_Expectancy_Data.csv")
        st.success("Dataset carregado com sucesso!")
    except FileNotFoundError:
        st.error("Erro: O arquivo 'Life_Expectancy_Data.csv' não foi encontrado.")
        st.stop()

    # Visualizar o dataset
    st.subheader("📋 Visualização do Dataset")
    st.dataframe(df.head())

    # Estatísticas descritivas
    st.subheader("📈 Estatísticas Descritivas")
    st.write(df.describe())

    # Gráfico de correlação
    st.subheader("📊 Correlação entre as Variáveis")
    corr = df.corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Mapa de Correlação")
    st.plotly_chart(fig_corr, use_container_width=True)

    # Gráfico de distribuição das variáveis
    st.subheader("📊 Distribuição das Taxas de Vacinação")
    for col in ['Hepatitis B', 'Polio', 'Diphtheria']:
        fig_dist = px.histogram(df, x=col, nbins=30, title=f"Distribuição de {col}")
        st.plotly_chart(fig_dist, use_container_width=True)

# Página de previsão
elif menu == "Previsão":
    st.title("📈 Previsão de Expectativa de Vida")
    st.write("Insira os dados abaixo para prever a expectativa de vida:")

    # Variáveis do modelo
    feature_columns = ['Hepatitis B', 'Polio', 'Diphtheria']

    # Validação da entrada do utilizador
    input_data = {}
    for feature in feature_columns:
        input_value = st.slider(f"{feature} (%)", 0.0, 100.0, 50.0)
        input_data[feature] = input_value

    # Validar entrada
    if not all(0 <= input_data[feature] <= 100 for feature in feature_columns):
        st.warning("Por favor, insira valores entre 0 e 100.")
        st.stop()

    # Criar H2OFrame para entrada do modelo
    input_df = pd.DataFrame([input_data])
    h2o_input = H2OFrame(input_df)

    # Fazer previsão
    if st.button("Fazer Previsão"):
        prediction = rf_model.predict(h2o_input).as_data_frame().iloc[0, 0]
        st.write(f"**Expectativa de Vida Prevista:** {round(prediction, 2)} anos")

# Página de previsão em massa
elif menu == "Previsão em Massa":
    st.title("📂 Previsão em Massa")
    st.write("Envie um arquivo CSV contendo os dados para prever a expectativa de vida para múltiplas entradas.")

    # Upload do arquivo
    uploaded_file = st.file_uploader("Envie o arquivo CSV", type=["csv"])
    if uploaded_file:
        try:
            mass_df = pd.read_csv(uploaded_file)
            st.success("Arquivo carregado com sucesso!")
            st.dataframe(mass_df.head())

            # Converter para H2OFrame
            h2o_mass_input = H2OFrame(mass_df)

            # Fazer previsões em massa
            predictions = rf_model.predict(h2o_mass_input).as_data_frame()
            mass_df['Life Expectancy Prediction'] = predictions

            # Exibir resultados
            st.subheader("📋 Previsões")
            st.dataframe(mass_df)

            # Download do arquivo com previsões
            csv = mass_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Baixar Resultados", data=csv, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")

# Encerrar o H2O ao sair
st.sidebar.write("🔌 **H2O está ativo. Lembre-se de encerrar ao terminar.**")
if st.sidebar.button("Encerrar H2O"):
    h2o.shutdown(prompt=False)
