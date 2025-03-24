# Criar script com dashboard em Streamlit
streamlit_code = """
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# === Funções de indicadores técnicos ===
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, short=12, long=26, signal=9):
    ema_short = series.ewm(span=short, adjust=False).mean()
    ema_long = series.ewm(span=long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_hist

def create_sequences(data, window_size):
    X = []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
    return np.array(X)

# === Interface Streamlit ===
st.title("📈 Previsão BTC/USD - Classificação por Rede Neural")

uploaded_file = st.file_uploader("Faça upload do arquivo CSV com dados de 1h do BTC/USDC", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, header=None)
    df.columns = [
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base Volume', 'Taker Buy Quote Volume', 'Ignore'
    ]
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='us')
    cols_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume',
                       'Quote Asset Volume', 'Taker Buy Base Volume', 'Taker Buy Quote Volume']
    df[cols_to_convert] = df[cols_to_convert].astype(float)

    # Calcular indicadores
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = compute_macd(df['Close'])

    # Remover NaNs
    df.dropna(inplace=True)

    # Normalizar
    features = df[['Close', 'RSI', 'MACD']]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # Criar sequência
    window_size = 30
    X = create_sequences(scaled_features, window_size)

    # Carregar modelo treinado
    try:
        model = load_model("modelo_btc_classificador.h5")
        preds = model.predict(X)
        pred_labels = (preds > 0.5).astype(int)

        # Mostrar gráfico
        st.subheader("🔍 Previsões nas últimas 100 janelas")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(pred_labels[-100:], marker='o', label='Previsão')
        ax.set_title("Classe prevista: 1 = Alta, 0 = Queda")
        ax.set_xlabel("Tempo (últimas janelas)")
        ax.set_ylabel("Classe")
        ax.legend()
        st.pyplot(fig)

        st.success("Previsão concluída com sucesso!")

    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
"""

# Salvar script do dashboard
streamlit_path = "/mnt/data/dashboard_btc_classificador.py"
with open(streamlit_path, "w") as f:
    f.write(streamlit_code)

streamlit_path