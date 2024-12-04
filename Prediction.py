import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib


def load_data(disease):
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
    chart_data = df[['DT_NOTIFIC', 'TAXA_INC_MES']]
    chart_data['DT_NOTIFIC'] = pd.to_datetime(chart_data['DT_NOTIFIC'])
    chart_data['Month'] = chart_data['DT_NOTIFIC'].dt.to_period('M')
    chart_data = chart_data.groupby('Month').mean().reset_index()
    st.bar_chart(chart_data.set_index('Month')['TAXA_INC_MES'])


def get_streamlit_background_color():
    return "#0e1117"  

def plot_bar_chart(df, disease):
    chart_data = df[['DT_NOTIFIC', 'TAXA_INC_MES', 'SURTO']].copy()
    chart_data['DT_NOTIFIC'] = pd.to_datetime(chart_data['DT_NOTIFIC'])
    chart_data['Month'] = chart_data['DT_NOTIFIC'].dt.to_period('M')
    chart_data = chart_data.groupby('Month').mean().reset_index()

    background_color = get_streamlit_background_color()

    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)

    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    ax.bar(chart_data['Month'].astype(str), chart_data['TAXA_INC_MES'], color='#84c1eb', edgecolor='none')
    outbreaks = chart_data[chart_data['SURTO'] == 1]
    ax.scatter(outbreaks['Month'].astype(str), outbreaks['TAXA_INC_MES'], color='#ff4d4d', s=100, zorder=5)

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#ffffff')
    ax.set_title(f'Monthly Incidence Rate - {disease}', color='white', fontsize=16, pad=20)
    ax.set_xlabel('', color='white', fontsize=12)
    ax.set_ylabel('Incidence Rate', color='white', fontsize=12)
    ax.tick_params(axis='x', colors='white', labelsize=10, rotation=45)
    ax.tick_params(axis='y', colors='white', labelsize=10)
    plt.tight_layout()

    st.pyplot(fig)


def load_model_for_disease(disease):
    model_path = f'models/{disease.lower()}.h5'
    model = load_model(model_path)
    return model


def predict_cases(disease, X_test, y_test_regress, y_test_class):
    model = load_model_for_disease(disease)

    y_pred_regress = model.predict(X_test)

    return y_pred_regress



def load_csv(disease):
    if disease == "Dengue":
        df = pd.read_csv('data/dengue/df_separado.csv', parse_dates=['DT_NOTIFIC'])
    elif disease == "Chikungunya":
        df = pd.read_csv('data/chikungunya/df_separado.csv', parse_dates=['DT_NOTIFIC'])
    elif disease == "Zika":
        df = pd.read_csv('data/zika/df_separado.csv', parse_dates=['DT_NOTIFIC'])
    
    return df


def prediction():
    st.markdown('>Prediction')
    st.header("Infectious Disease Outbreak Predictions")
    
    disease = st.selectbox('Select the disease', ['Dengue', 'Chikungunya', 'Zika'])
    
    X_train, X_test, y_train_class, y_test_class, y_train_regress, y_test_regress = load_data(disease)
    df = load_csv(disease)
    
    if st.checkbox(f"View {disease} outbreak graph"):
        plot_bar_chart(df, disease)
    
    if st.button(f"Predicting {disease} outbreaks in 2023"):
        model_path = f"models/{disease.lower()}.h5"
        model = load_model(model_path)

        test_results = model.evaluate(X_test, {'classification': y_test_class, 'regression': y_test_regress})
        #st.write(f"Test results: {test_results}")

        scaler_path = f"data/{disease.lower()}/scalers.pkl"
        scaler_dict = joblib.load(scaler_path)
        scaler_X = scaler_dict.get('scaler')
        scaler_y = scaler_dict.get('scaler_y')

        y_pred = model.predict(X_test)

        if isinstance(y_pred, list) and len(y_pred) == 2:
            y_pred_class = y_pred[0]  
            y_pred_reg = y_pred[1]  
            
            if len(y_pred_class.shape) == 2 and y_pred_class.shape[1] == 1:
                y_pred_class = y_pred_class.flatten()  # Flatten to one dimension

            y_pred_class = (y_pred_class > 0.5).astype(int)

            if len(y_pred_reg.shape) == 2 and y_pred_reg.shape[1] > 1:
                y_pred_reg = y_pred_reg[:, 0]  # Flatten to one dimension if needed

            if len(y_pred_reg.shape) == 2:
                y_pred_reg = y_pred_reg.flatten()

            y_test_reg_desescalonado = scaler_y.inverse_transform(y_test_regress.reshape(-1, 1)).flatten()
            y_pred_reg_desescalonado = scaler_y.inverse_transform(y_pred_reg.reshape(-1, 1)).flatten()

            y_pred_reg_desescalonado = np.maximum(y_pred_reg_desescalonado, 0)

            # Set plot style to match the background color of the site
            background_color = get_streamlit_background_color()

            fig, ax = plt.subplots(figsize=(14, 8))
            fig.patch.set_facecolor(background_color)
            ax.set_facecolor(background_color)

            timestamp_test = np.array(df['DT_NOTIFIC'][-len(y_test_regress):])
            ax.plot(timestamp_test, y_test_reg_desescalonado, label='Actual (Number of Cases)', color='#84c1eb', linewidth=2)
            ax.plot(timestamp_test, y_pred_reg_desescalonado, label='Predicted (Number of Cases)', color='#ffa726', linewidth=2)

            surtos_previstos = timestamp_test[y_pred_class >= 0.9]
            surtos_valores = y_pred_reg_desescalonado[y_pred_class >= 0.9]
            ax.scatter(surtos_previstos, surtos_valores, color='#ff4d4d', label='Predicted Outbreak', s=50)

            ax.set_title('Predictions vs Actual Data', fontsize=16, color='white')
            ax.set_xlabel('Date', fontsize=12, color='white')
            ax.set_ylabel('Number of Cases', fontsize=12, color='white')
            ax.tick_params(axis='x', colors='white', labelsize=10, rotation=45)
            ax.tick_params(axis='y', colors='white', labelsize=10)
            ax.legend(fontsize=12)
            ax.grid(True, color='white', linestyle='--', linewidth=0.5)

            st.pyplot(fig)
        else:
            st.error("The prediction output is not in the expected format. Please check the model architecture.")