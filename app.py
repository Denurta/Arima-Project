import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Judul dashboard
st.title('Dashboard Peramalan ARIMA')

# Upload file CSV
uploaded_file = st.file_uploader("Pilih file CSV", type="csv")

if uploaded_file is not None:
    try:
        # Membaca file CSV
        df = pd.read_csv(uploaded_file)

        # Pilih kolom numerik
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if numeric_columns.empty:
            st.error("File CSV tidak memiliki kolom numerik untuk peramalan.")
        else:
            col = st.selectbox('Pilih kolom data deret waktu:', numeric_columns)
            data = pd.to_numeric(df[col], errors='coerce').dropna()

            # Konversi ke time series jika ada kolom waktu
            time_col = st.selectbox('Pilih kolom waktu (opsional):', [None] + list(df.columns), index=0)
            if time_col:
                try:
                    df[time_col] = pd.to_datetime(df[time_col])
                    data.index = df[time_col]
                except Exception as e:
                    st.warning(f"Gagal memproses kolom waktu: {e}")

            # 1. Identifikasi (Plot data dan uji stasioneritas)
            st.subheader("1. Identifikasi")

            # Plot data asli
            st.write("Plot Data Asli")
            st.line_chart(data)

            # Uji Stasioneritas (ADF Test)
            st.write("Uji Stasioneritas (ADF Test)")
            adf_result = adfuller(data)
            st.write(f"ADF Statistic: {adf_result[0]}")
            st.write(f"P-Value: {adf_result[1]}")

            if adf_result[1] > 0.05:
                st.warning("Data tidak stasioner. Pertimbangkan pembedaan (d).")

            # Plot ACF dan PACF
            st.write("Plot ACF dan PACF")
            fig_acf, ax_acf = plt.subplots()
            plot_acf(data, ax=ax_acf)
            st.pyplot(fig_acf)

            fig_pacf, ax_pacf = plt.subplots()
            plot_pacf(data, ax=ax_pacf)
            st.pyplot(fig_pacf)

            # 2. Estimasi parameter ARIMA
            st.subheader("2. Estimasi")
            p = st.number_input('Masukkan nilai p (Autoregressive):', min_value=0, value=1, step=1)
            d = st.number_input('Masukkan nilai d (Pembedaan):', min_value=0, value=1, step=1)
            q = st.number_input('Masukkan nilai q (Moving Average):', min_value=0, value=1, step=1)

            # Lakukan estimasi otomatis setelah upload file
            try:
                # Fit model ARIMA
                model = ARIMA(data, order=(p, d, q))
                model_fit = model.fit()

                # 3. Cek Diagnostik
                st.subheader("3. Cek Diagnostik")
                st.write("Ringkasan Model")
                st.text(model_fit.summary())

                # Plot residuals
                residuals = model_fit.resid
                st.write("Plot Residual")
                fig_resid, ax_resid = plt.subplots()
                ax_resid.plot(residuals)
                ax_resid.axhline(y=0, color='red', linestyle='--')
                st.pyplot(fig_resid)

                # Statistik Diagnostik
                st.write("Statistik Residual")
                st.write(residuals.describe())

                # 4. Peramalan
                st.subheader("4. Peramalan")
                steps = st.number_input('Langkah peramalan ke depan:', min_value=1, value=10, step=1)
                freq = st.selectbox('Pilih frekuensi data:', ['D - Harian', 'W - Mingguan', 'M - Bulanan', 'A - Tahunan'])
                freq_code = freq.split('-')[0].strip()

                # Lakukan peramalan
                forecast = model_fit.forecast(steps=steps)

                # Atur indeks tanggal untuk hasil peramalan
                if isinstance(data.index, pd.DatetimeIndex):
                    last_date = data.index[-1]
                    forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq=freq_code)
                else:
                    forecast_index = range(len(data), len(data) + steps)

                # Tampilkan hasil peramalan
                st.write("Hasil Peramalan:")
                forecast_df = pd.DataFrame({
                    'Tanggal': forecast_index,
                    'Forecast': forecast
                })
                st.dataframe(forecast_df)

                # Visualisasi hasil peramalan
                st.write("Plot Peramalan")
                fig_forecast, ax_forecast = plt.subplots()
                ax_forecast.plot(data, label='Data Asli')
                ax_forecast.plot(forecast_index, forecast, label='Peramalan', color='red')
                ax_forecast.legend()
                st.pyplot(fig_forecast)

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
    except Exception as e:
        st.error(f"Gagal memproses file CSV: {e}")
else:
    st.write("Silakan upload file CSV untuk melanjutkan.")
