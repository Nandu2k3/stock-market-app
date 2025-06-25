


import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image

# Streamlit Page Config
st.set_page_config(layout="centered", page_title="Stock Market Visualizer")

# Title and description
st.title("📊 Stock Market Visualizer")
st.markdown("This app lets you visualize the closing price of any stock using different chart types.")

# Input section
ticker = st.text_input("🔤 Enter Stock Ticker Symbol (e.g., AAPL, TSLA)", "AAPL")

graph_type = st.radio("📈 Select Graph Type", ["Line", "Scatter", "Bar"])

bg_image_file = st.file_uploader("🖼️ Optional: Upload Background Image", type=["jpg", "jpeg", "png"])

# Fetch data when ticker is entered
if ticker:
    try:
        data = yf.download(ticker)

        if not data.empty:
            fig, ax = plt.subplots(figsize=(12, 6))

            if graph_type.lower() == 'line':
                ax.plot(data['Close'], label='Close Price', color='blue')
            elif graph_type.lower() == 'scatter':
                ax.scatter(data.index, data['Close'], label='Close Price', color='green')
            elif graph_type.lower() == 'bar':
                ax.bar(data.index, data['Close'], label='Close Price', color='orange')

            ax.set_title(f'Closing Price of {ticker}', fontsize=16)
            ax.set_xlabel('Date')
            ax.set_ylabel('Closing Price')
            ax.legend()

            # Optional background image overlay
            if bg_image_file is not None:
                img = Image.open(bg_image_file)
                ax.imshow(img, extent=ax.axis(), aspect='auto', alpha=0.2)

            st.pyplot(fig)
        else:
            st.warning("⚠️ No data found. Please check the ticker symbol.")
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")# thisi is my python code 


