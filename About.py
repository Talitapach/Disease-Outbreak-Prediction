import streamlit as st
import numpy as np
import pandas as pd
import pickle


st.markdown('>About Project')
st.header("Predicting Infectious Disease Outbreaks Using Recurrent Neural Networks with Multivariate Data from DataSUS of Pernambuco")
#st.image('img.jpg')
st.subheader('Introduction')
st.sidebar.markdown('- Introduction')
st.write('Infectious diseases, especially arboviruses such as dengue, chikungunya, and Zika, represent a significant public health challenge in vulnerable regions such as Pernambuco. In 2022, the state recorded 18,200 cases of dengue, 16,800 of chikungunya, and 322 of Zika, highlighting the urgent need for effective strategies to predict outbreaks. This study uses Recurrent Neural Networks (RNNs), specifically the Long Short-Term Memory (LSTM) model, to develop a prediction system based on historical data from DataSUS.')
st.write('')
st.sidebar.markdown('- Methodology')
st.subheader('Methodology')
# Informações sobre a base de dados
st.write("**Database**: SINAN data (2017-2023) related to dengue, chikungunya, and Zika.")

# Informações sobre o pré-processamento
st.write("**Preprocessing**: Handling inconsistencies, normalization, calculation of incidence rates to define an outbreak threshold, and data preparation for modeling.")

# Informações sobre os modelos
st.write(
    "**Models**:\n"
    "- Simple LSTM: 2 layers (1 LSTM + 1 Dense).\n"
    "- Bidirectional LSTM: 3 layers (1 BiLSTM + 2 Densely Connected)."
)

# Informações sobre o treinamento
st.write(
    "**Training**:\n"
    "- Hyperparameter tuning with Keras Tuner.\n"
    "- Cross-validation (5 folds).\n"
    "- Early stopping."
)

# Informações sobre os tamanhos de janela
st.write("**Windows**: Tested window sizes of 60, 90, and 120 steps.")

# Informações sobre as métricas
st.write(
    "**Metrics**:\n"
    "- Regression: MAPE and MedAPE.\n"
    "- Classification: F1-Score and AUC-ROC."
)

st.image('utils\Processo.png')
