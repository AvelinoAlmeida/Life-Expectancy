import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar a página
st.set_page_config(page_title="Sistema de Apoio à Decisão", layout="wide")

def load_css(file_path):
    """
    Carrega um arquivo CSS e aplica no Streamlit.

    Args:
        file_path (str): Caminho para o arquivo CSS.
    """
    try:
        with open(file_path, "r") as f:
            css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Erro: Arquivo CSS '{file_path}' não encontrado.")
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o CSS: {e}")

# Carregar o arquivo CSS geral
load_css("./css/config.css")


# Carregar os modelos salvos
try:
    socioeconomic_model = joblib.load("./models/socioeconomic_predictions.pkl")
    vaccination_model = joblib.load("./models/vaccination_predictions.pkl")
    st.sidebar.success("Modelos carregados com sucesso!")
except FileNotFoundError:
    st.sidebar.error("Erro: Um ou mais modelos não foram encontrados. Verifique o diretório './models'.")
    st.stop()


# Configurar o menu no Streamlit
st.sidebar.title("Menu")
menu = st.sidebar.radio(
    "Ir para:",
    ["Home", "Estatísticas", "Treino", "Socioeconômico", "Vacinação"],
    index=0
)


# Controle de navegação
if menu == "Home":
    st.title("🏠 Bem-vindo!")
elif menu == "Estatísticas":
    st.title("📊 Análises e Estatísticas do Dataset")
elif menu == "Treino":
    st.title("📈 Treino de Modelos")    
elif menu == "Socioeconômico":
    st.title("🌍 Modelo Socioeconômico")
elif menu == "Vacinação":
    st.title("💉 Modelo de Vacinação")



## Página Home
if menu == "Home":

    st.write("Esta é a página inicial do sistema de apoio à decisão.")
    
    # Adicionar uma imagem
    st.image("./img/imagem1.jpg", caption="Sistema de Apoio à Decisão")

    st.markdown("<h2>🌍 Previsão de Expectativa de Vida</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p>Este sistema ajuda a prever a expectativa de vida com base em variáveis demográficas, clínicas e socioeconômicas.</p>",
        unsafe_allow_html=True
    )


# Carregar o dataset original
    df = pd.read_csv("./data/Life Expectancy Data.csv")
    
# Estatísticas gerais do dataset original
    st.subheader("Dataset Original")
    st.write(df.describe())

  # Dados em falta
    st.subheader("Dados em falta")
    st.image("./img/grafico1.png", caption="Dados em falta")
    
# Matriz de Correlação do Dataset Original
    st.subheader("Correlação - Dataset Original")
    numeric_df = df.select_dtypes(include=["number"])
    corr_matrix = numeric_df.corr()
    st.write(corr_matrix.style.background_gradient(cmap="coolwarm"))

# Análise dos atributos
    st.subheader("Análise dos atributos")
    st.image("./img/grafico2.png", caption="Análise dos atributos")


# Página de análises
elif menu == "Estatísticas":
    
# Adicionar uma imagem ilustrativa
    st.image("./img/imagem2.jpg", caption="Processo de Análise")

# Introdução
    st.subheader("📋 Observações sobre o novo Dataset")
    st.markdown("""
    - **Os dados foram limpos** e todas as linhas com valores nulos foram removidas.
    - **Não existe mais valores nulos no dataset.**
    - O **dataset original** tinha **2.938 registos** e **22 colunas**.
    - Após a limpeza, o **dataset atual** contém **1.649 entradas**.
    """)
# Distribuição dos Dados
    st.subheader("Análise dos atributos depois de limpos")
    st.image("./img/grafico3.png", caption="Análise dos atributos")

# GDP
    st.subheader("📋 Aplicação da Transformação Logarítmica")
    st.markdown("""
    Técnica de pré-processamento de dados em que se aplica a função logarítmica a um ou mais valores de um conjunto de dados
    quando os dados apresentam uma distribuição assimétrica positiva (longa cauda à direita), 
    a transformação logarítmica ajuda a tornar a distribuição mais simétrica
    """)    
    st.image("./img/grafico4.png", caption="Análise dos atributos")

# Distribuição Final dos Dados
    st.subheader("Distribuição Final dos Dados")
    st.image("./img/grafico5.png", caption="Dataset_Clean")
  


# Página de Treino de Modelos
elif menu == "Treino":
    st.write("Nesta página, apresentamos os modelos de previsão treinados, suas métricas de desempenho e análises visuais.")

    import json

# Carregar métricas do modelo Socioeconômico
    try:
        with open("./models/socioeconomic_metrics.json", "r") as f:
            socioeconomic_metrics = json.load(f)
    except FileNotFoundError:
        socioeconomic_metrics = None
        st.error("Métricas do modelo Socioeconômico não encontradas!")

# Carregar métricas do modelo de Vacinação
    try:
        with open("./models/vaccination_metrics.json", "r") as f:
            vaccination_metrics = json.load(f)
    except FileNotFoundError:
        vaccination_metrics = None
        st.error("Métricas do modelo de Vacinação não encontradas!")

    st.subheader("Objetivo - Previsão média de vida (anos)")
    st.image("./img/grafico6.png", caption="Dataset_Clean")



# Apresentar os resultados do modelo Socioeconômico
    st.subheader("🌍 Modelo Socioeconômico (Random Forest)")

    if socioeconomic_metrics:
        # Métricas do modelo Socioeconômico
        st.markdown("**Métricas Iniciais de Teste do Modelo Socioeconômico**")
        st.write(f"- **MAE:** {socioeconomic_metrics.get('mae', 'N/A')}")
        st.write(f"- **MSE:** {socioeconomic_metrics.get('mse', 'N/A')}")
        st.write(f"- **R²:** {socioeconomic_metrics.get('r2', 'N/A')}")

        st.image("./img/grafico7.png", caption="Modelo Teste Socioeconômico")
        st.image("./img/grafico8.png", caption="Modelo Final - Gradient Boosting Machine")


# Apresentar os resultados do modelo de Vacinação
    st.subheader("💉 Modelo de Vacinação (Random Forest)")

    if vaccination_metrics:
        # Métricas do modelo de Vacinação
        st.markdown("**Métricas do Modelo de Vacinação**")
        st.write(f"- **MAE:** {vaccination_metrics.get('mae', 'N/A')}")
        st.write(f"- **MSE:** {vaccination_metrics.get('mse', 'N/A')}")
        st.write(f"- **R²:** {vaccination_metrics.get('r2', 'N/A')}")

        st.image("./img/grafico9.png", caption="Modelo Teste Vacinação")
        st.image("./img/grafico10.png", caption="Modelo Final - Gradient Boosting Machine")

        

## Página de previsão para o modelo Socioeconômico
elif menu == "Socioeconômico":
    st.title("📈 Previsão de Expectativa de Vida")
    st.write("Ajuste os valores abaixo para prever automaticamente a expectativa de vida com base em alterações percentuais nas variáveis socioeconômicas:")

    # Carregar o dataset limpo para referência
    try:
        df_clean = pd.read_csv("./data/Life_Expectancy_Clean.csv")
        st.success("Dataset limpo carregado com sucesso!")
    except FileNotFoundError:
        st.error("Erro: O dataset './data/Life_Expectancy_Clean.csv' não foi encontrado.")
        st.stop()

    # Variáveis do modelo socioeconômico (usadas no treinamento)
    feature_columns = [
        'percentage expenditure', 
        'Total expenditure', 
        'GDP', 
        'Income composition of resources', 
        'Schooling'
    ]

    # Calcular os valores médios das variáveis no dataset limpo
    avg_values = df_clean[feature_columns].mean()

    # Criar sliders para ajustes percentuais com base no valor médio
    st.subheader("📊 Ajuste Percentual das Variáveis (em relação ao valor médio)")
    input_data = {}
    for feature in feature_columns:
        # Valor médio da variável
        base_value = avg_values[feature]

        # Slider para ajuste percentual (-10% a +10%, com passos de 2%)
        adjustment = st.slider(
            f"Ajuste de {feature} (%)",
            -10, 10, 0, step=2,  # Intervalo de ajuste: -10% a +10%, começando em 0
            key=feature
        )

        # Calcular o valor ajustado com base no percentual
        adjusted_value = base_value * (1 + adjustment / 100)
        input_data[feature] = adjusted_value

        # Mostrar o valor ajustado
        st.write(f"Valor ajustado de **{feature}**: {adjusted_value:.2f} (base: {base_value:.2f}, ajuste: {adjustment}%)")

    # Criar DataFrame para entrada do modelo
    input_df = pd.DataFrame([input_data])

    # Adicionar colunas ausentes com valores padrão (0.0) para garantir compatibilidade
    for column in socioeconomic_model.feature_names_in_:
        if column not in input_df.columns:
            input_df[column] = 0.0

    # Garantir a mesma ordem de colunas usada no treinamento
    input_df = input_df[socioeconomic_model.feature_names_in_]

    # Fazer previsão automaticamente ao alterar os sliders
    try:
        prediction = socioeconomic_model.predict(input_df)[0]
        st.success(f"**Expectativa de Vida Prevista:** {round(prediction, 2)} anos")
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
    st.title("📈 Previsão de Expectativa de Vida")
    st.write("Ajuste os valores abaixo para prever automaticamente a expectativa de vida com base em taxas de vacinação:")

    # Carregar o dataset limpo para referência
    try:
        df_clean = pd.read_csv("./data/Life_Expectancy_Clean.csv")
        st.success("Dataset limpo carregado com sucesso!")
    except FileNotFoundError:
        st.error("Erro: O dataset './data/Life_Expectancy_Clean.csv' não foi encontrado.")
        st.stop()

    # Variáveis do modelo de vacinação
    feature_columns = [
        'Hepatitis B',
        'Polio',
        'Diphtheria',
        'percentage expenditure'
    ]

    # Calcular os valores médios das variáveis no dataset limpo
    avg_values = df_clean[feature_columns].mean()

    # Criar sliders para ajustes percentuais com base no valor médio
    st.subheader("📊 Ajuste Percentual das Variáveis (em relação ao valor médio)")
    input_data = {}
    for feature in feature_columns:
        # Valor médio da variável
        base_value = avg_values[feature]

        # Slider para ajuste percentual (-10% a +10%, com passos de 2%)
        adjustment = st.slider(
            f"Ajuste de {feature} (%)",
            -10, 10, 0, step=2,  # Intervalo de ajuste: -10% a +10%
            key=feature
        )

        # Calcular o valor ajustado com base no percentual
        adjusted_value = base_value * (1 + adjustment / 100)
        input_data[feature] = adjusted_value

        # Mostrar o valor ajustado
        st.write(f"Valor ajustado de **{feature}**: {adjusted_value:.2f} (base: {base_value:.2f}, ajuste: {adjustment}%)")

    # Criar DataFrame para entrada do modelo
    input_df = pd.DataFrame([input_data])

    # Adicionar colunas ausentes com valores padrão (0.0) para garantir compatibilidade
    for column in vaccination_model.feature_names_in_:
        if column not in input_df.columns:
            input_df[column] = 0.0

    # Garantir a mesma ordem de colunas usada no treinamento
    input_df = input_df[vaccination_model.feature_names_in_]

    # Fazer previsão automaticamente ao alterar os sliders
    try:
        prediction = vaccination_model.predict(input_df)[0]
        st.success(f"**Expectativa de Vida Prevista (Vacinação):** {round(prediction, 2)} anos")
    except ValueError as e:
        st.error(f"Erro ao fazer a previsão: {e}")

    # Exibir a importância das variáveis
    st.subheader("📊 Importância das Variáveis no Modelo de Vacinação")
    try:
        feature_importances = pd.DataFrame({
            "Variável": vaccination_model.feature_names_in_,
            "Importância": vaccination_model.feature_importances_
        }).sort_values(by="Importância", ascending=False)
        st.bar_chart(feature_importances.set_index("Variável"))
    except AttributeError:
        st.warning("A importância das variáveis não está disponível para o modelo carregado.")
