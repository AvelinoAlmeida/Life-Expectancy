import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


import streamlit as st

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

# Configurar o menu no Streamlit
st.sidebar.title("Menu")
menu = st.sidebar.radio(
    "Ir para:",
    ["Início", "Estatísticas", "Modelos", "Socioeconômico", "Vacinação"],
    index=0
)


# Controle de navegação
if menu == "Início":
    st.title("🏠 Bem-vindo!")
    st.write("Esta é a página inicial do sistema de apoio à decisão.")
    
    # Adicionar uma imagem
    st.image("./img/imagem1.jpg", caption="Sistema de Apoio à Decisão")

elif menu == "Estatísticas":
    st.title("📊 Estatísticas e Análises")
    st.write("Aqui apresentamos análises descritivas dos dados.")
elif menu == "Modelos":
    st.title("📈 Treino de Modelos")    
elif menu == "Socioeconômico":
    st.title("🌍 Modelo Socioeconômico")
elif menu == "Vacinação":
    st.title("💉 Modelo de Vacinação")


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
    st.title("📊 Análises e Estatísticas do Dataset")
    
    # Adicionar uma imagem ilustrativa
    st.image("./img/imagem2.jpg", caption="Processo de Análise")

    # Carregar datasets para análises
    try:
        # Carregar o dataset original
        df = pd.read_csv("./data/Life Expectancy Data.csv")
        st.success("Dataset Original carregado com sucesso!")
        
        # Estatísticas gerais do dataset original
        st.subheader("📊 Estatísticas do Dataset Original")
        st.write(df.describe())

        # Matriz de Correlação do Dataset Original
        st.subheader("📊 Matriz de Correlação - Dataset Original")
        numeric_df = df.select_dtypes(include=["number"])
        corr_matrix = numeric_df.corr()
        st.write(corr_matrix.style.background_gradient(cmap="coolwarm"))

        # Gráficos de dispersão - Dataset Original
        st.subheader("📈 Gráficos de Dispersão - Dataset Original")
        selected_columns = [
            "percentage expenditure", "GDP", 
            "Income composition of resources", "Schooling"
        ]
        for column in selected_columns:
            if column.strip() in df.columns:
                st.write(f"### Relação entre **{column.strip()}** e **Expectativa de Vida**")
                plt.figure(figsize=(5, 5))
                sns.scatterplot(x=df[column.strip()], y=df["Life expectancy "])
                plt.title(f"Relação entre {column.strip()} e Expectativa de Vida")
                plt.xlabel(column.strip())
                plt.ylabel("Expectativa de Vida")
                st.pyplot(plt)
                plt.clf()
            else:
                st.warning(f"Coluna '{column.strip()}' não encontrada no dataset.")

        # Visualizar a distribuição de valores nulos - Dataset Original
        st.subheader("📉 Distribuição de Valores Nulos - Dataset Original")
        if df.isnull().sum().sum() > 0:
            plt.figure(figsize=(12, 6))
            sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
            plt.title("Distribuição de Valores Nulos no Dataset Original", fontsize=16)
            plt.xlabel("Variáveis")
            plt.ylabel("Registos")
            st.pyplot(plt)
            st.write("### Soma de valores nulos por variável:")
            st.write(df.isnull().sum())
        else:
            st.success("Não existem valores nulos no Dataset Original!")
        
        # Carregar o dataset limpo
        df_clean = pd.read_csv("./data/Life_Expectancy_Clean.csv")
        st.success("Dataset Limpo carregado com sucesso!")
        
        # Estatísticas gerais do dataset limpo
        st.subheader("📊 Estatísticas do Dataset Limpo")
        st.write(df_clean.describe())
        
        # Matriz de Correlação do Dataset Limpo
        st.subheader(" Matriz de Correlação - Dataset Limpo")
        clean_corr_matrix = df_clean.corr()
        st.write(clean_corr_matrix.style.background_gradient(cmap="coolwarm"))

        # Histograma das variáveis - Dataset Limpo
        st.subheader("📊 Distribuição das Variáveis - Dataset Limpo")

        try:
            # Gerar histogramas para todas as variáveis numéricas no dataset limpo
            numeric_columns = df_clean.select_dtypes(include=["number"]).columns

            # Configurar o tamanho da figura e criar histogramas
            fig, ax = plt.subplots(figsize=(10, 5))
            df_clean[numeric_columns].hist(
                bins=20,
                color="skyblue",
                alpha=0.7,
                ax=ax,
                grid=False,
            )
            plt.suptitle("📈 Distribuição das Variáveis no Dataset Limpo", fontsize=16)
            plt.tight_layout()

            # Exibir os histogramas no Streamlit
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Ocorreu um erro ao gerar os histogramas: {e}")

    except FileNotFoundError as e:
        st.error(f"Erro: {e}")


# Página de Treino de Modelos
elif menu == "Modelos":
    st.title("📊 Modelos de Treino")
    st.write("Nesta página, apresentamos os modelos de previsão treinados, suas métricas de desempenho e análises visuais.")

    import json
    import matplotlib.pyplot as plt
    import seaborn as sns

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

    # Apresentar os resultados do modelo Socioeconômico
    st.subheader("🌍 Modelo Socioeconômico (Random Forest)")

    if socioeconomic_metrics:
        # Métricas do modelo Socioeconômico
        st.markdown("**Métricas do Modelo Socioeconômico**")
        st.write(f"- **MAE:** {socioeconomic_metrics.get('mae', 'N/A')}")
        st.write(f"- **MSE:** {socioeconomic_metrics.get('mse', 'N/A')}")
        st.write(f"- **R²:** {socioeconomic_metrics.get('r2', 'N/A')}")

        # Gráfico de Importância das Variáveis - Modelo Socioeconômico
        st.markdown("**📊 Importância das Variáveis - Modelo Socioeconômico**")
        feature_importance_rf = socioeconomic_metrics.get("feature_importance", {})
        if feature_importance_rf:
            features_rf = list(feature_importance_rf.keys())
            importances_rf = list(feature_importance_rf.values())

            # Criar gráfico de barras
            fig_rf, ax_rf = plt.subplots(figsize=(8, 5))
            sns.barplot(x=importances_rf, y=features_rf, palette="Blues_d", ax=ax_rf)
            ax_rf.set_title("Importância das Variáveis (Random Forest)")
            ax_rf.set_xlabel("Importância")
            ax_rf.set_ylabel("Variáveis")
            st.pyplot(fig_rf)
        else:
            st.warning("Importância das variáveis não disponível para o modelo Socioeconômico.")

        # Gráfico de Valores Reais vs. Previstos - Modelo Socioeconômico
        st.markdown("**📈 Valores Reais vs. Previstos - Modelo Socioeconômico**")
        y_test_rf = socioeconomic_metrics.get("y_test", [])
        y_pred_rf = socioeconomic_metrics.get("y_pred", [])
        if y_test_rf and y_pred_rf:
            fig_scatter_rf, ax_scatter_rf = plt.subplots(figsize=(8, 5))
            sns.scatterplot(x=y_test_rf, y=y_pred_rf, ax=ax_scatter_rf, alpha=0.6)
            ax_scatter_rf.plot([min(y_test_rf), max(y_test_rf)], [min(y_test_rf), max(y_test_rf)], 'r--')
            ax_scatter_rf.set_title("Valores Reais vs. Previstos (Random Forest)")
            ax_scatter_rf.set_xlabel("Valores Reais")
            ax_scatter_rf.set_ylabel("Valores Previstos")
            st.pyplot(fig_scatter_rf)
        else:
            st.warning("Dados de comparação (Valores Reais vs. Previstos) não disponíveis para o modelo Socioeconômico.")

    st.markdown("---")

    # Apresentar os resultados do modelo de Vacinação
    st.subheader("💉 Modelo de Vacinação (Gradient Boosting Machine)")

    if vaccination_metrics:
        # Métricas do modelo de Vacinação
        st.markdown("**Métricas do Modelo de Vacinação**")
        st.write(f"- **MAE:** {vaccination_metrics.get('mae', 'N/A')}")
        st.write(f"- **MSE:** {vaccination_metrics.get('mse', 'N/A')}")
        st.write(f"- **R²:** {vaccination_metrics.get('r2', 'N/A')}")

        # Gráfico de Importância das Variáveis - Modelo de Vacinação
        st.markdown("**📊 Importância das Variáveis - Modelo de Vacinação**")
        feature_importance_gbm = vaccination_metrics.get("feature_importance", {})
        if feature_importance_gbm:
            features_gbm = list(feature_importance_gbm.keys())
            importances_gbm = list(feature_importance_gbm.values())

            # Criar gráfico de barras
            fig_gbm, ax_gbm = plt.subplots(figsize=(8, 5))
            sns.barplot(x=importances_gbm, y=features_gbm, palette="Greens_d", ax=ax_gbm)
            ax_gbm.set_title("Importância das Variáveis (Gradient Boosting Machine)")
            ax_gbm.set_xlabel("Importância")
            ax_gbm.set_ylabel("Variáveis")
            st.pyplot(fig_gbm)
        else:
            st.warning("Importância das variáveis não disponível para o modelo de Vacinação.")

        

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
