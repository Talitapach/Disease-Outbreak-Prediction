import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler




# Função para carregar dados com base na escolha da doença
def load_data(disease):
    # Ajuste para carregar os dados .npy para cada doença conforme já foi discutido
    if disease == "Dengue":
        X_train = np.load('data/dengue/X_train.npy')
        X_test = np.load('data/dengue/X_test.npy')
        y_train_class = np.load('data/dengue/y_train_class.npy')
        y_test_class = np.load('data/dengue/y_test_class.npy')
        y_train_regress = np.load('data/dengue/y_train_regress.npy')
        y_test_regress = np.load('data/dengue/y_test_regress.npy')
    elif disease == "Chikungunya":
        X_train = np.load('data/chikungunya/X_train.npy')
        X_test = np.load('data/chikungunya/X_test.npy')
        y_train_class = np.load('data/chikungunya/y_train_class.npy')
        y_test_class = np.load('data/chikungunya/y_test_class.npy')
        y_train_regress = np.load('data/chikungunya/y_train_regress.npy')
        y_test_regress = np.load('data/chikungunya/y_test_regress.npy')
    elif disease == "Zika":
        X_train = np.load('data/zika/X_train.npy')
        X_test = np.load('data/zika/X_test.npy')
        y_train_class = np.load('data/zika/y_train_class.npy')
        y_test_class = np.load('data/zika/y_test_class.npy')
        y_train_regress = np.load('data/zika/y_train_regress.npy')
        y_test_regress = np.load('data/zika/y_test_regress.npy')

    return X_train, X_test, y_train_class, y_test_class, y_train_regress, y_test_regress



def plot_bar_chart_bonito(df, disease):
    # Preparamos o gráfico de barras com a Taxa de Incidência por mês
    chart_data = df[['DT_NOTIFIC', 'TAXA_INC_MES']]
    chart_data['DT_NOTIFIC'] = pd.to_datetime(chart_data['DT_NOTIFIC'])
    
    # Agrupar por mês e calcular a média da Taxa de Incidência
    chart_data['Month'] = chart_data['DT_NOTIFIC'].dt.to_period('M')
    chart_data = chart_data.groupby('Month').mean().reset_index()

    # Exibir o gráfico
    st.bar_chart(chart_data.set_index('Month')['TAXA_INC_MES'])


# Captura o tema atual (precisa que o modo tema esteja ativado no Streamlit)
def get_streamlit_background_color():
    return "#0e1117"  # Substitua por um código hexadecimal padrão ou dinâmico do seu site.

# Função para criar o gráfico com as bolinhas vermelhas
def plot_bar_chart(df, disease):
    chart_data = df[['DT_NOTIFIC', 'TAXA_INC_MES', 'SURTO']].copy()
    chart_data['DT_NOTIFIC'] = pd.to_datetime(chart_data['DT_NOTIFIC'])
    chart_data['Month'] = chart_data['DT_NOTIFIC'].dt.to_period('M')
    chart_data = chart_data.groupby('Month').mean().reset_index()

    # Detectar cor do fundo do site
    background_color = get_streamlit_background_color()

    # Criar gráfico
    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)

    # Definir o fundo com a cor dinâmica
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    # Criar barras e adicionar os surtos
    ax.bar(chart_data['Month'].astype(str), chart_data['TAXA_INC_MES'], color='#84c1eb', edgecolor='none')
    outbreaks = chart_data[chart_data['SURTO'] == 1]
    ax.scatter(outbreaks['Month'].astype(str), outbreaks['TAXA_INC_MES'], color='#ff4d4d', s=100, zorder=5)

    # Personalizar visualização
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#ffffff')
    ax.set_title(f'Taxa de Incidência Mensal - {disease}', color='white', fontsize=16, pad=20)
    ax.set_xlabel('', color='white', fontsize=12)
    ax.set_ylabel('Taxa de Incidência', color='white', fontsize=12)
    ax.tick_params(axis='x', colors='white', labelsize=10, rotation=45)
    ax.tick_params(axis='y', colors='white', labelsize=10)
    plt.tight_layout()

    # Exibir no Streamlit
    st.pyplot(fig)


# Função para carregar o modelo correto com base na escolha da doença
def load_model_for_disease(disease):
    model_path = f'models/{disease.lower()}.h5'
    model = load_model(model_path)
    return model


def predict_cases(disease, X_test, y_test_regress, y_test_class):
    model = load_model_for_disease(disease)

    # Fazer previsões
    y_pred_regress = model.predict(X_test)

    # Aqui você pode comparar com os valores reais ou salvar as previsões
    return y_pred_regress


# Função para carregar o CSV correspondente
def load_csv(disease):
    # Definir o caminho correto do CSV com base na doença
    if disease == "Dengue":
        df = pd.read_csv('data/dengue/df_separado.csv', parse_dates=['DT_NOTIFIC'])
    elif disease == "Chikungunya":
        df = pd.read_csv('data/chikungunya/df_separado.csv', parse_dates=['DT_NOTIFIC'])
    elif disease == "Zika":
        df = pd.read_csv('data/zika/df_separado.csv', parse_dates=['DT_NOTIFIC'])
    
    return df

# Página 1: Explicação do trabalho
def page_1():
    st.title("Previsão de Surtos de Doenças Infecciosas em Recife")
    st.write("""
        Este projeto tem como objetivo a previsão de surtos de doenças infecciosas como dengue, chikungunya e zika, utilizando modelos de LSTM para identificar padrões nos dados históricos de casos registrados em Recife. 
        A previsão visa antecipar surtos para melhorar a resposta de saúde pública e a alocação de recursos.
    """)

# Página 2: Explicação sobre doenças infecciosas
def page_2():
    st.title("Doenças Infecciosas: Dengue, Chikungunya e Zika")
    st.write("""
        As doenças como dengue, chikungunya e zika são causadas por vírus transmitidos por mosquitos, e têm mostrado surtos recorrentes em várias regiões do Brasil, especialmente em Recife.
        - **Dengue**: Causada pelo vírus da dengue, é uma doença febril que pode causar dor muscular e nas articulações.
        - **Chikungunya**: Causada por um vírus transmitido pelo mosquito Aedes aegypti, causando febre e dores nas articulações.
        - **Zika**: Também transmitida pelo mosquito Aedes aegypti, pode causar febre, erupções cutâneas e malformações congênitas quando contraída por gestantes.
    """)
    

# Página 3: Exibição de surtos e previsão
def page_3():
    st.title("Previsões de Surtos de Doenças Infecciosas")
    
    # Seleção da doença
    disease = st.selectbox('Selecione a doença', ['Dengue', 'Chikungunya', 'Zika'])
    
    # Carregar os dados e CSV
    X_train, X_test, y_train_class, y_test_class, y_train_regress, y_test_regress = load_data(disease)
    df = load_csv(disease)
    
    # Exibir gráfico de surtos
    if st.checkbox(f"Exibir gráfico de surtos de {disease}"):
        plot_bar_chart(df, disease)
    
    # Botão de previsão para 2023
    if st.button(f"Prever surtos de {disease} em 2023"):
        model_path = f"models/{disease.lower()}.h5"
        model = load_model(model_path)

        # Avaliação do modelo
        test_results = model.evaluate(X_test, {'classification': y_test_class, 'regression': y_test_regress})
        st.write(f"Resultados no teste: {test_results}")

        scaler = MinMaxScaler()
        #scaler.fit(X_train)
        #joblib.dump(scaler, 'scalers/disease_scaler.pkl')

        # Previsões
        y_pred = model.predict(X_test)
    
        # Verificando e ajustando as previsões
        if isinstance(y_pred, list) and len(y_pred) == 2:
            y_pred_class = y_pred[0]  # A previsão de classificação
            y_pred_reg = y_pred[1]    # A previsão de regressão
            
            # Confirmação da forma das previsões
            print("Forma de y_pred_class:", y_pred_class.shape)
            print("Forma de y_pred_reg:", y_pred_reg.shape)
            
            # Verifique e ajuste a saída da classificação
            if len(y_pred_class.shape) == 2 and y_pred_class.shape[1] == 1:
                y_pred_class = y_pred_class.flatten()  # Achata para uma dimensão

            # Classificação binária
            y_pred_class = (y_pred_class > 0.5).astype(int)

            # Verifique se a saída da regressão está em um formato esperado (use a primeira coluna, se necessário)
            if len(y_pred_reg.shape) == 2 and y_pred_reg.shape[1] > 1:
                y_pred_reg = y_pred_reg[:, 0]  # Seleciona apenas a primeira coluna para a regressão

            # Certifique-se de que y_pred_reg seja um array 1D após a seleção
            if len(y_pred_reg.shape) == 2:
                y_pred_reg = y_pred_reg.flatten()
        else:
            st.error("A saída da previsão não está no formato esperado. Verifique a arquitetura do modelo.")
            return  # Para de executar se o formato não estiver correto
        

        # Ajuste do scaler para inversão da escala
        scaler = joblib.load('scalers/disease_scaler.pkl')

        # Transformação inversa e achatar
        y_test_reg_desescalonado = scaler.inverse_transform(y_test_regress.reshape(-1, 1)).flatten()
        y_pred_reg_desescalonado = scaler.inverse_transform(y_pred_reg.reshape(-1, 1)).flatten()

        # Forçar valores negativos para zero (se aplicável)
        y_pred_reg_desescalonado = np.maximum(y_pred_reg_desescalonado, 0)


        # Plotar os dados reais e previstos
        fig, ax = plt.subplots(figsize=(14, 8))
        timestamp_test = np.array(df['timestamp'][-len(y_test_regress):])  # Supondo que haja uma coluna de timestamps
        ax.plot(timestamp_test, y_test_reg_desescalonado, label='Real (Número de Casos)', color='blue', linewidth=2)
        ax.plot(timestamp_test, y_pred_reg_desescalonado, label='Previsto (Número de Casos)', color='orange', linewidth=2)

        # Destaque para surtos previstos
        surtos_previstos = timestamp_test[y_pred_class >= 0.9]  # Timestamps onde há surtos previstos
        surtos_valores = y_pred_reg_desescalonado[y_pred_class >= 0.9]  # Valores previstos para surtos
        ax.scatter(surtos_previstos, surtos_valores, color='red', label='Surto Previsto', s=50)

        # Configuração do gráfico
        ax.set_title('Previsões x Dados Reais', fontsize=16)
        ax.set_xlabel('Data', fontsize=12)
        ax.set_ylabel('Número de Casos', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True)
        st.pyplot(fig)
        
    

# Navegação entre as páginas
def main():
    st.sidebar.title("Menu")
    page = st.sidebar.radio("Escolha a página", ["Explicação do Trabalho", "Doenças Infecciosas", "Previsões de Surtos"])
    
    if page == "Explicação do Trabalho":
        page_1()
    elif page == "Doenças Infecciosas":
        page_2()
    elif page == "Previsões de Surtos":
        page_3()

if __name__ == "__main__":
    main()
