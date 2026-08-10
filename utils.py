"""
utils.py
Berisi dua bagian utama:
  1. extract_features(image_path) -> mengekstrak 5 fitur numerik dari foto buah
     memakai Pillow + numpy (ringan, tidak butuh OpenCV/TensorFlow di HP).
  2. FruitClassifier -> memuat bobot model Logistic Regression (fruit_model.json)
     hasil training di model/train_model.py, lalu melakukan inferensi
     (softmax(W·x + b)) murni dengan numpy. Tidak butuh scikit-learn saat
     runtime, sehingga jauh lebih ringan saat di-build jadi APK.
"""

import json
import os
import numpy as np
from PIL import Image


def _rgb_to_hue(r, g, b):
    """Konversi vectorized RGB (0-255) -> Hue (0-360 derajat)."""
    r_, g_, b_ = r / 255.0, g / 255.0, b / 255.0
    max_c = np.maximum(np.maximum(r_, g_), b_)
    min_c = np.minimum(np.minimum(r_, g_), b_)
    delta = max_c - min_c
    delta_safe = np.where(delta == 0, 1e-6, delta)

    hue = np.zeros_like(max_c)
    is_r = (max_c == r_) & (delta != 0)
    is_g = (max_c == g_) & (delta != 0)
    is_b = (max_c == b_) & (delta != 0)

    hue = np.where(is_r, ((g_ - b_) / delta_safe) % 6, hue)
    hue = np.where(is_g, ((b_ - r_) / delta_safe) + 2, hue)
    hue = np.where(is_b, ((r_ - g_) / delta_safe) + 4, hue)
    hue = hue * 60.0
    hue = np.where(hue < 0, hue + 360, hue)
    return hue


def extract_features(image_path, resize_to=100):
    """
    Mengembalikan dict fitur:
        brightness, saturation, dark_ratio, brown_ratio, variance
    Definisi fitur ini SAMA PERSIS dengan yang dipakai saat training
    di model/train_model.py supaya hasil prediksi konsisten.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((resize_to, resize_to))
    arr = np.asarray(img).astype(np.float32)  # shape (H, W, 3)

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    brightness_map = (r + g + b) / 3.0

    max_c = np.max(arr, axis=-1)
    min_c = np.min(arr, axis=-1)
    saturation_map = np.where(max_c > 0, (max_c - min_c) / np.maximum(max_c, 1e-6), 0)

    brightness = float(np.mean(brightness_map))
    saturation = float(np.mean(saturation_map))

    dark_ratio = float(np.mean(brightness_map < 80))

    # Bercak coklat/busuk: hue di rentang coklat-oranye tua (10-45 derajat),
    # cenderung gelap dan TIDAK terlalu saturated (merah/oranye segar biasanya
    # saturasi tinggi, jadi tidak ikut ke-flag sebagai "bercak").
    hue_map = _rgb_to_hue(r, g, b)
    brown_mask = (
        (hue_map >= 10) & (hue_map <= 45)
        & (brightness_map < 130)
        & (saturation_map < 0.55)
    )
    brown_ratio = float(np.mean(brown_mask))

    variance = float(np.std(brightness_map))

    return {
        "brightness": brightness,
        "saturation": saturation,
        "dark_ratio": dark_ratio,
        "brown_ratio": brown_ratio,
        "variance": variance,
    }


class FruitClassifier:
    """Wrapper inferensi model Logistic Regression yang sudah dilatih."""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "model", "fruit_model.json")
        with open(model_path, "r") as f:
            data = json.load(f)

        self.feature_names = data["feature_names"]
        self.class_names = data["class_names"]
        self.mean = np.array(data["scaler_mean"], dtype=np.float64)
        self.scale = np.array(data["scaler_scale"], dtype=np.float64)
        self.coef = np.array(data["coef"], dtype=np.float64)          # (3, 5)
        self.intercept = np.array(data["intercept"], dtype=np.float64)  # (3,)
        self.accuracy = data.get("accuracy")

    def _softmax(self, z):
        z = z - np.max(z)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def predict(self, features: dict):
        x = np.array([features[name] for name in self.feature_names], dtype=np.float64)
        x_scaled = (x - self.mean) / self.scale

        logits = self.coef @ x_scaled + self.intercept
        probs = self._softmax(logits)

        idx = int(np.argmax(probs))
        label = self.class_names[idx]
        confidence = float(probs[idx])
        prob_dict = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}

        return label, confidence, prob_dict, features

    def predict_image(self, image_path):
        features = extract_features(image_path)
        return self.predict(features)