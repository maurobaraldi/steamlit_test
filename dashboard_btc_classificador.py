# Atualizar o código do dashboard para buscar dados da Binance em tempo real
streamlit_realtime_code = """
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import requests

# === Indicadores técnicos ===
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

def get_binance_data(symbol='BTCUSDT', interval='1h', limit=200):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Asset Volume', 'Number of Trades',
        'Taker Buy Base Volume', 'Taker Buy Quote Volume', 'Ignore'
    ])
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(float)
    return df

# === Streamlit UI ===
st.title("📡 Previsão BTC/USDT em Tempo Real")

if st.button("🔄 Obter e Prever Dados em Tempo Real"):
    df = get_binance_data()

    # Indicadores técnicos
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'] = compute_macd(df['Close'])

    df.dropna(inplace=True)

    # Normalização
    features = df[['Close', 'RSI', 'MACD']]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # Criar janelas
    window_size = 30
    X = create_sequences(scaled_features, window_size)

    try:
        model = load_model("modelo_btc_classificador.h5")
        preds = model.predict(X)
        pred_labels = (preds > 0.5).astype(int)

        # Última previsão
        ultima_classe = int(pred_labels[-1][0])
        st.metric(label="📊 Última Previsão", value="Alta 📈" if ultima_classe == 1 else "Queda 📉")

        # Gráfico
        st.subheader("📈 Histórico das últimas 100 previsões")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(pred_labels[-100:], marker='o', label='Previsão')
        ax.set_title("1 = Alta, 0 = Queda")
        ax.set_xlabel("Tempo")
        ax.set_ylabel("Classe Prevista")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
"""