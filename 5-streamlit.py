import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Configurar o Streamlit
st.set_page_config(page_title="Previsão de Expectativa de Vida", layout="wide")

# Barra lateral de navegação
st.sidebar.title("Menu de Navegação")
menu = st.sidebar.radio("Ir para", ["Início", "Estatísticas", "Previsão"])

# Carregar o dataset
try:
    df = pd.read_csv("./data/Life Expectancy Data.csv")
    st.success("Dataset carregado com sucesso!")
except FileNotFoundError:
    st.error("Erro: O arquivo 'life_expectancy_data.csv' não foi encontrado.")
    st.stop()

# Página inicial (Início)
if menu == "Início":
    st.markdown("<h2>🌍 Previsão de Expectativa de Vida</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p>Este sistema ajuda a prever a expectativa de vida com base em variáveis demográficas e clínicas.</p>",
        unsafe_allow_html=True
    )

# Página de análises
elif menu == "Estatísticas":
    st.title("📊 Estatísticas e Análises dos Dados")

    # Verificar se a coluna 'Age' existe no dataset
    if 'Age' not in df.columns:
        st.warning("A coluna 'Age' não foi encontrada no dataset. Criando uma coluna simulada...")
        if 'Birth_Year' in df.columns:
            # Exemplo de criação de coluna 'Age' a partir de 'Birth_Year'
            df['Age'] = 2024 - df['Birth_Year']
        else:
            # Caso nenhuma alternativa esteja disponível, criar uma coluna fictícia para demonstração
            df['Age'] = np.random.randint(0, 100, size=len(df))
            st.warning("Uma coluna fictícia de 'Age' foi criada para demonstração.")

    # Criar grupos de idade
    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['0-20', '21-40', '41-60', '61-80', '80+']
    )

    # Contar frequências por grupo de idade
    age_life_counts = df['Age_Group'].value_counts().reset_index()
    age_life_counts.columns = ['Faixa Etária', 'Frequência']

    # Criar gráfico interativo
    fig = px.bar(
        age_life_counts,
        x="Faixa Etária",
        y="Frequência",
        labels={"Faixa Etária": "Faixa Etária", "Frequência": "Frequência"},
        title="Distribuição da Expectativa de Vida por Faixa Etária"
    )

    # Exibir o gráfico no Streamlit
    st.subheader("Distribuição da Expectativa de Vida por Faixa Etária")
    st.plotly_chart(fig, use_container_width=True)

    # Exibir a tabela resultante
    st.write("Tabela de Frequências por Faixa Etária:")
    st.dataframe(age_life_counts)

# Página de previsão
elif menu == "Previsão":
    st.title("📈 Previsão de Expectativa de Vida")
    st.write("Insira os dados abaixo para prever a expectativa de vida:")

    # Variáveis do modelo
    feature_columns = ["'Hepatitis B', 'Polio', 'Diphtheria'"]

    # Captura de entrada do utilizador
    input_data = {}
    for feature in feature_columns:
        input_data[feature] = st.slider(f"{feature}", 0.0, 100.0, 50.0)

    # Criar DataFrame para entrada do modelo
    input_df = pd.DataFrame([input_data])

    # Simular previsão (substituir pelo modelo treinado)
    if st.button("Fazer Previsão"):
        prediction = np.random.uniform(50, 85)  # Substituir com `model.predict(input_df)`
        st.write(f"**Expectativa de Vida Prevista:** {round(prediction, 2)} anos")
