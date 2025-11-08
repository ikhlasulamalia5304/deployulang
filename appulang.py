import streamlit as st
import pandas as pd
import joblib 
import numpy as np
from xgboost import XGBClassifier # Wajib di-import untuk memuat model JSON

# --- 1. MUAT MODEL (Gunakan format JSON yang stabil) ---
# PASTIKAN FILE model_xgb.json DAN scaler.pkl SUDAH ADA DI FOLDER LOKAL
MODEL_FILE = 'model_xgb.json' 
SCALER_FILE = 'scaler.pkl'

try:
    # 1. Muat Scaler (Joblib aman untuk Scikit-learn scaler)
    scaler = joblib.load(SCALER_FILE)
    
    # 2. Muat Model XGBoost (Gunakan format native JSON)
    model = XGBClassifier() # Inisiasi objek kosong
    model.load_model(MODEL_FILE) # Muat data model dari file JSON

    # *** PERBAIKAN ATTRIBUTE ERROR UNTUK predict_proba() ***
    # Ini diperlukan karena serialisasi versi lama/baru tidak menyertakan atribut ini.
    # Kita tahu ini adalah masalah klasifikasi biner (2 kelas: 0 dan 1).
    if not hasattr(model, 'n_classes_'):
        model.n_classes_ = 2 
    if not hasattr(model, 'classes_'):
        model.classes_ = np.array([0, 1])

    st.sidebar.success("Model dan scaler berhasil dimuat!") 
except FileNotFoundError:
    st.error(f"Error: Pastikan file model ({MODEL_FILE}) dan scaler ({SCALER_FILE}) sudah di-push.")
    st.stop()
except Exception as e:
    st.error(f"Error saat memuat model: Terjadi masalah kompatibilitas model. {e}")
    st.stop()

# --- 2. TITTLE DAN DESKRIPSI APLIKASI ---
st.set_page_config(page_title="Prediksi Anemia", layout="centered")
st.title("🩸 Prediksi Klasifikasi Anemia dengan XGBoost")
st.markdown("---")
st.write("**Model ini menggunakan fitur rasio Hemoglobin/MCV (Hb/MCV Ratio) untuk meningkatkan akurasi, sesuai dengan riset.**")

# --- 3. INPUT DARI PENGGUNA ---

st.header("Input Data Pasien")

# Mendapatkan input dari pengguna
col1, col2 = st.columns(2)

with col1:
    hemoglobin = st.number_input('1. Hemoglobin (HGB, g/dL)', min_value=0.0, max_value=20.0, value=12.0)
    mch = st.number_input('2. MCH (pg)', min_value=0.0, max_value=100.0, value=27.0)
    mcv = st.number_input('3. MCV (fL)', min_value=0.0, max_value=150.0, value=85.0)

with col2:
    # FITUR MCHC DITAMBAHKAN DI SINI UNTUK MEMENUHI 6 FITUR SCALER
    mchc = st.number_input('4. MCHC (g/dL)', min_value=0.0, max_value=50.0, value=33.0) 
    gender = st.selectbox('5. Jenis Kelamin', ['Wanita', 'Pria'])

# Konversi Gender ke 0 (Wanita) atau 1 (Pria) - Asumsi Label Encoding Anda
gender_encoded = 1 if gender == 'Pria' else 0


# --- 4. PREDIKSI (Saat tombol ditekan) ---
if st.button('Prediksi Risiko Anemia', type="primary"):
    
    # 4a. FEATURE ENGINEERING: Hitung Hb/MCV Ratio (Wajib!)
    try:
        # Rasio Hb/MCV (HGB / MCV)
        hb_mcv_ratio = hemoglobin / mcv
    except ZeroDivisionError:
        st.error("MCV tidak boleh nol.")
        st.stop()
    
    # Kumpulkan input pengguna sesuai URUTAN FITUR PELATIHAN
    # URUTAN: [Gender (Encoded), HGB, MCH, MCV, MCHC, Hb/MCV Ratio] -> 6 Fitur
    data_input_list = [gender_encoded, hemoglobin, mch, mcv, mchc, hb_mcv_ratio] 
    
    # Konversi ke NumPy array (scaler butuh format 2D)
    data_input = np.array([data_input_list])
    
    # 4b. SCALING DATA: Terapkan scaler
    # Scaling dilakukan pada SEMUA 6 fitur
    data_input_scaled = scaler.transform(data_input)
    
    # 4c. Lakukan Prediksi
    prediksi = model.predict(data_input_scaled) 
    prediksi_prob = model.predict_proba(data_input_scaled)
    
    # 4d. Tampilkan Hasil
    st.subheader("Hasil Prediksi Model:")
    
    prob_anemia = prediksi_prob[0][1] * 100
    prob_normal = prediksi_prob[0][0] * 100
    
    if prediksi[0] == 1:
        st.error(f"🚨 ANEMIA: Risiko tinggi terdeteksi.")
        st.markdown(f"**Probabilitas Anemia:** `{prob_anemia:.2f}%`")
        st.markdown(f"Probabilitas Normal: `{prob_normal:.2f}%`")
    else:
        st.success(f"✅ TIDAK ANEMIA: Risiko rendah.")
        st.markdown(f"**Probabilitas Normal:** `{prob_normal:.2f}%`")
        st.markdown(f"Probabilitas Anemia: `{prob_anemia:.2f}%`")
    
    st.markdown("---")
    st.caption("Aplikasi ini dibuat menggunakan model XGBoost dengan Feature Engineering rasio Hb/MCV.")