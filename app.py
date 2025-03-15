import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
import matplotlib.pyplot as plt
from PIL import Image


# Fungsi untuk memproses data
@st.cache_data
def preprocessing():
    df = pd.read_csv('Anaemic_Data.csv')
    
    # Encoding kolom Anaemic dan Sex
    labelencoder = LabelEncoder()
    df['Anaemic_encode'] = labelencoder.fit_transform(df['Anaemic'])
    df['Sex_encode'] = df['Sex'].apply(lambda x: 1 if x.strip().upper() == 'M' else 0)

    return df


# Fungsi untuk membangun model
@st.cache_resource
def build_model(df):
    regr = linear_model.LogisticRegression()
    X = df[['%Red Pixel', '%Blue pixel', '%Green pixel', 'Hb', 'Sex_encode']]
    y = df[['Anaemic_encode']]
    
    regr.fit(X, y.values.ravel())  # Ravel digunakan agar y memiliki dimensi yang sesuai
    return regr


# Fungsi untuk melakukan prediksi
def predict_anemia(regr, input_data):
    prediction = regr.predict([input_data])
    return "Anaemic" if prediction[0] == 1 else "Not Anaemic"


# --- Tampilan Streamlit ---
st.set_page_config(page_title="Anaemia Prediction App", layout="wide")

# Menampilkan gambar atau logo
image = Image.open("medical_logo.png")  # Ganti dengan gambar yang sesuai
st.image(image, width=200)

# Header aplikasi
st.title('🔬 Anemia Prediction App')
st.write("Aplikasi ini menggunakan **Machine Learning** untuk memprediksi apakah seseorang mengalami anemia berdasarkan karakteristik darah mereka.")

# Sidebar
st.sidebar.header("🔢 Input Data")
st.sidebar.write("Silakan masukkan nilai berikut untuk mendapatkan prediksi:")

# Memuat data dan model
df = preprocessing()
regr = build_model(df)

# Menampilkan contoh data
st.subheader("📊 Contoh Data")
st.dataframe(df.head())

# Input pengguna melalui sidebar
red_pixel = st.sidebar.slider("🔴 % Red Pixel", 0.0, 100.0, 50.0)
blue_pixel = st.sidebar.slider("🔵 % Blue Pixel", 0.0, 100.0, 50.0)
green_pixel = st.sidebar.slider("🟢 % Green Pixel", 0.0, 100.0, 50.0)
hb = st.sidebar.number_input("💉 Hemoglobin Level (Hb)", min_value=0.0, max_value=20.0, value=13.5)
sex = st.sidebar.radio("🚻 Jenis Kelamin", ["Male", "Female"])
sex_encoded = 1 if sex == "Male" else 0

# Tombol untuk melakukan prediksi
if st.sidebar.button("🔍 Prediksi Anemia"):
    input_data = [red_pixel, blue_pixel, green_pixel, hb, sex_encoded]
    result = predict_anemia(regr, input_data)

    # Menampilkan hasil prediksi
    st.subheader("📌 Hasil Prediksi")
    st.write(f"**Prediksi:** {result}")
    
    # Warna latar belakang hasil
    if result == "Anaemic":
        st.error("⚠️ Orang ini kemungkinan mengalami Anemia.")
    else:
        st.success("✅ Orang ini tidak mengalami Anemia.")

# Visualisasi Data
st.subheader("📈 Distribusi Hemoglobin")
fig, ax = plt.subplots()
df["Hb"].hist(bins=20, alpha=0.7, color="blue", edgecolor="black", ax=ax)
ax.set_xlabel("Level Hb")
ax.set_ylabel("Frekuensi")
ax.set_title("Distribusi Hemoglobin pada Dataset")
st.pyplot(fig)
