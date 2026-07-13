import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from PIL import Image
import io

# KONFIGURASI - menyesuaikan path
MODEL_PATH = os.path.join('model', 'animal_classifier_densenet121.keras')
CLASS_MAP_PATH = os.path.join('model', 'class_names_id.json')
IMG_SIZE = (224, 224)


app = Flask(__name__)
CORS(app)  # frontend bisa akses API ini

print("Memuat model, mohon tunggu...")
model = load_model(MODEL_PATH)

with open(CLASS_MAP_PATH, 'r', encoding='utf-8') as f:
    class_names_id = json.load(f)

# Urutan kelas HARUS sama pada folder saat training
class_labels_it = sorted(class_names_id.keys())

print("Model berhasil dimuat!")
print("Kelas:", class_labels_it)


def preprocess_image(img_bytes):
    """Ubah bytes gambar upload menjadi array siap prediksi."""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/health', methods=['GET'])
def health():
    """Endpoint sederhana untuk cek backend hidup."""
    return jsonify({"status": "ok", "message": "Backend klasifikasi hewan aktif"})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint utama: terima file gambar, kembalikan hasil prediksi.

    Request: multipart/form-data dengan field 'image'
    Response JSON contoh:
    {
        "success": true,
        "prediction": {
            "label_it": "gatto",
            "label_id": "Kucing",
            "confidence": 96.42
        },
        "top3": [
            {"label_id": "Kucing", "confidence": 96.42},
            {"label_id": "Anjing", "confidence": 2.10},
            {"label_id": "Tupai", "confidence": 0.85}
        ]
    }
    """
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "Tidak ada file gambar yang diupload"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "Nama file kosong"}), 400

    try:
        img_bytes = file.read()
        img_array = preprocess_image(img_bytes)

        predictions = model.predict(img_array)[0]

        # Ambil top-3 prediksi tertinggi
        top_indices = predictions.argsort()[-3:][::-1]

        top3 = []
        for idx in top_indices:
            label_it = class_labels_it[idx]
            label_id = class_names_id.get(label_it, label_it)
            confidence = float(predictions[idx]) * 100
            top3.append({"label_id": label_id, "confidence": round(confidence, 2)})

        best = top3[0]
        top_idx = int(top_indices[0])

        return jsonify({
            "success": True,
            "prediction": {
                "label_it": class_labels_it[top_idx],
                "label_id": best["label_id"],
                "confidence": best["confidence"]
            },
            "top3": top3
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)