import streamlit as st
import joblib
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
from fpdf import FPDF
import base64

# Load model dan encoder
model = joblib.load("ensemble_model.pkl")
le_dict = joblib.load("label_encoders.pkl")
features = joblib.load("features.pkl")

# Mapping nama kolom ke Bahasa Indonesia
label_mapping = {
    "Gender": "Jenis Kelamin",
    "Age": "Umur (±100)",
    "Occupation": "Pekerjaan",
    "Sleep Duration": "Durasi Tidur (jam, ±24)",
    "Quality of Sleep": "Kualitas Tidur (±10)",
    "Physical Activity Level": "Tingkat Aktivitas Fisik (±10)",
    "Stress Level": "Tingkat Stres (±10)",
    "BMI Category": "Kategori BMI",
    "Blood Pressure": "Tekanan Darah",
    "Heart Rate": "Detak Jantung (40–150 bpm)",
    "Daily Steps": "Jumlah Langkah Harian (±30000)",
    "Sleep Disorder": "Gangguan Tidur"
}

# Kolom yang tidak perlu ditampilkan
excluded_columns = ['Person ID']

# Fungsi membuat file PDF
def generate_pdf(nama, hasil, solusi):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Hasil Prediksi Gangguan Tidur", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Nama Pengguna: {nama}")
    pdf.multi_cell(0, 10, f"Hasil Prediksi: {hasil}")

    # Hapus emoji agar tidak error saat menyimpan PDF
    solusi_clean = solusi.encode('ascii', 'ignore').decode()
    pdf.multi_cell(0, 10, f"\nSolusi:\n{solusi_clean}")

    pdf_path = "hasil_prediksi.pdf"
    pdf.output(pdf_path)
    return pdf_path

def get_download_link(file_path, label="📄 Download Hasil PDF"):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="{file_path}">{label}</a>'
    return href

# Login sederhana
def login():
    st.subheader("🔒 Login Pengguna")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state["login"] = True
        else:
            st.error("Login gagal!")

# Halaman Home
def home():
    st.title("🛌 Prediksi Gangguan Tidur")
    st.markdown("Aplikasi ini memprediksi gangguan tidur berdasarkan gaya hidup dan kesehatan.")

# Halaman Prediksi
def prediction():
    st.subheader("📋 Input Data Pengguna")
    nama_pengguna = st.text_input("Nama Lengkap")
    input_data = []

    for col in features:
        if col in excluded_columns:
            continue
        label = label_mapping.get(col, col)
        if col in le_dict:
            le = le_dict[col]
            options = le.classes_.tolist()
            selected = st.selectbox(label, options)
            encoded = le.transform([selected])[0]
            input_data.append(encoded)
        else:
            if col == "Sleep Duration":
                val = st.number_input(label, min_value=0.0, step=0.1)
            else:
                val = st.number_input(label, min_value=0, step=1, format="%d")
            input_data.append(val)

    if st.button("🔍 Prediksi"):
        data_array = np.array(input_data).reshape(1, -1)
        pred = model.predict(data_array)[0]

        # Decode Label prediksi
        if 'Sleep Disorder' in le_dict:
            label_decoder = le_dict['Sleep Disorder']
            label = label_decoder.inverse_transform([pred])[0]
        else:
            label = pred

        if label == 'None':
            label = 'Normal'

        st.success(f"Prediksi Gangguan Tidur untuk **{nama_pengguna}**: **{label}**")

        saran = {
            "Insomnia": f"""
            **💡 Solusi untuk {nama_pengguna} (Insomnia):**

            1. Tidur dan bangun di waktu yang sama setiap hari.
            2. Hindari kafein, alkohol, dan nikotin setidaknya 6 jam sebelum tidur.
            3. Matikan gadget dan lampu terang minimal 1 jam sebelum tidur.
            4. Lakukan teknik relaksasi (misalnya pernapasan dalam, doa, atau journaling).
            5. Pastikan suhu dan pencahayaan kamar nyaman.
            6. Konsultasikan dengan psikolog jika gangguan berlangsung lebih dari 2 minggu.
            7. Hindari tidur siang terlalu lama (maks 20 sampai 30 menit).
            """,

            "Sleep Apnea": f"""
            **Solusi untuk {nama_pengguna} (Sleep Apnea):**

            1. Konsultasikan dengan dokter THT atau paru.
            2. Gunakan alat CPAP jika disarankan oleh dokter.
            3. Turunkan berat badan jika memiliki BMI tinggi.
            4. Hindari menyamping, bukan telentang.
            5. Hindari alkohol, rokok, dan obat penenang.
            6. Jaga pola makan sehat dan olahraga rutin.
            7. Pertimbangkan pemeriksaan tidur (Polysomnografi).
            """
        }

        if label in saran:
            solusi_text = saran[label]
            st.markdown(solusi_text)
            pdf_file = generate_pdf(nama_pengguna, label, solusi_text)
            st.markdown(get_download_link(pdf_file, unsafe_allow_html=True))

# Halaman Tentang
def about():
    st.subheader("📖 Tentang Aplikasi")
    st.write("Aplikasi ini dibuat menggunakan model ensemble (Random Forest + XGBoost) ")
    st.write("Model dilatih di Google Colab dan diimplementasikan dengan Streamlit.")



def main():
    if "login" not in st.session_state:
        st.session_state["login"] = False

    if not st.session_state["login"]:
        login()
    else:
        with st.sidebar:
            selected = option_menu("Main Menu", ["Home", "Prediksi", "Tentang", "Logout"],
                                   icons=['house', 'bar-chart', 'info-circle', 'box-arrow-left'],
                                   menu_icon="cast", default_index=0)

            if selected == "Home":
                home()
            elif selected == "Prediksi":
                prediction()
            elif selected == "Tentang":
                about()
            elif selected == "Logout":
                st.session_state["login"] = False
                st.rerun()
    return  # 👈 Diperlukan untuk mencegah eksekusi lanjutan

if __name__ == '__main__':
    main()
