# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


# Fungsi untuk mengubah gambar menjadi array dan melakukan prediksi
def classify_image(model, image, input_size):
    img_array = np.array(image.resize((input_size, input_size)))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    return predictions


# Fungsi utama untuk aplikasi Streamlit
def main():
    st.title("Aplikasi Klasifikasi Gambar")

    # Unggah gambar
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Tampilkan gambar yang diunggah
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang Diunggah", use_column_width=True)

        # Tombol untuk melakukan prediksi
        if st.button("Klasifikasikan"):
            # Muat model-model yang telah dilatih (gantilah dengan path yang sesuai)
            model_efficientnet = "my_model1_EfficientNetB0 (1).h5"
            model_vgg16 = "my_model2_vgg16.h5"
            model_xception = "best_model_Percobaan3_epoch 50.h5"

            model1 = tf.keras.models.load_model(model_efficientnet)
            model2 = tf.keras.models.load_model(model_vgg16)
            model3 = tf.keras.models.load_model(model_xception)

            # Lakukan prediksi untuk setiap model
            predictions_efficientnet = classify_image(model1, image, 224)
            predictions_vgg16 = classify_image(model2, image, 128)
            predictions_xception = classify_image(model3, image, 128)

            # Tampilkan hasil prediksi
            st.subheader("Hasil Klasifikasi:")
            st.text("EfficientNetB0:")
            class_names_efficientnet = [
                "Daun keriting",
                "Sehat",
            ]  # Ganti dengan nama kelas sesuai model Anda
            predicted_class_efficientnet = class_names_efficientnet[
                np.argmax(predictions_efficientnet)
            ]

            st.text("Kelas: " + predicted_class_efficientnet)
            st.json(
                {
                    "class_probabilities": {
                        "Daun keriting": float(predictions_efficientnet[0][0]),
                        "Sehat": float(predictions_efficientnet[0][1]),
                    }
                }
            )

            st.text("VGG16:")
            class_names_vgg16 = [
                "Daun keriting",
                "Sehat",
            ]  # Ganti dengan nama kelas sesuai model Anda
            predicted_class_vgg16 = class_names_vgg16[np.argmax(predictions_vgg16)]

            st.text("Kelas: " + predicted_class_vgg16)
            st.json(
                {
                    "class_probabilities": {
                        "Daun keriting": float(predictions_vgg16[0][0]),
                        "Sehat": float(predictions_vgg16[0][1]),
                    }
                }
            )

            st.text("Xception:")
            class_names_xception = [
                "Daun keriting",
                "Sehat",
            ]  # Ganti dengan nama kelas sesuai model Anda
            predicted_class_xception = class_names_xception[
                np.argmax(predictions_xception)
            ]

            st.text("Kelas: " + predicted_class_xception)
            st.json(
                {
                    "class_probabilities": {
                        "Daun keriting": float(predictions_xception[0][0]),
                        "Sehat": float(predictions_xception[0][1]),
                    }
                }
            )


# Jalankan aplikasi
if __name__ == "__main__":
    main()
