"""
train_model.py
Melatih model Machine Learning (Logistic Regression multi-kelas) untuk
mengklasifikasikan kualitas buah berdasarkan 5 fitur citra:
    1. brightness   -> rata-rata kecerahan (0-255)
    2. saturation   -> rata-rata saturasi warna (0-1)
    3. dark_ratio   -> proporsi piksel gelap (0-1)
    4. brown_ratio  -> proporsi piksel kecoklatan (0-1)
    5. variance     -> variasi warna permukaan (0-50)

Kelas target: 0 = Segar, 1 = Sedang, 2 = Busuk

Model hasil training disimpan sebagai fruit_model.json (bobot + bias + parameter
normalisasi) — bukan .pkl — supaya bisa di-load di HP tanpa perlu library
scikit-learn/scipy ter-install di Android (yang menyulitkan proses build APK).
Saat inferensi di aplikasi, prediksi cukup dihitung dengan operasi matriks
sederhana (softmax(W·x + b)) memakai numpy saja.

Jalankan: python train_model.py
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

RANDOM_STATE = 42
N_SAMPLES_PER_CLASS = 800

FEATURE_NAMES = ["brightness", "saturation", "dark_ratio", "brown_ratio", "variance"]
CLASS_NAMES = ["Segar", "Sedang", "Busuk"]


def generate_synthetic_dataset(n_per_class=N_SAMPLES_PER_CLASS, seed=RANDOM_STATE):
    """
    Membuat dataset sintetis yang meniru distribusi fitur citra buah nyata
    untuk 3 kondisi kualitas. Nilai-nilai ini didasarkan pada pengamatan umum:
    buah segar cenderung cerah & saturasi tinggi dengan sedikit bercak gelap,
    sedangkan buah busuk cenderung gelap dengan banyak bercak kecoklatan.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []

    # Kelas 0: Segar
    n = n_per_class
    brightness = rng.normal(175, 18, n).clip(60, 255)
    saturation = rng.normal(0.62, 0.10, n).clip(0, 1)
    dark_ratio = rng.normal(0.05, 0.03, n).clip(0, 1)
    brown_ratio = rng.normal(0.04, 0.03, n).clip(0, 1)
    variance = rng.normal(10, 4, n).clip(0, 50)
    X.append(np.stack([brightness, saturation, dark_ratio, brown_ratio, variance], axis=1))
    y.append(np.zeros(n))

    # Kelas 1: Sedang
    brightness = rng.normal(140, 20, n).clip(40, 255)
    saturation = rng.normal(0.45, 0.12, n).clip(0, 1)
    dark_ratio = rng.normal(0.18, 0.07, n).clip(0, 1)
    brown_ratio = rng.normal(0.16, 0.07, n).clip(0, 1)
    variance = rng.normal(20, 6, n).clip(0, 50)
    X.append(np.stack([brightness, saturation, dark_ratio, brown_ratio, variance], axis=1))
    y.append(np.ones(n))

    # Kelas 2: Busuk
    brightness = rng.normal(100, 22, n).clip(20, 255)
    saturation = rng.normal(0.28, 0.12, n).clip(0, 1)
    dark_ratio = rng.normal(0.38, 0.10, n).clip(0, 1)
    brown_ratio = rng.normal(0.36, 0.10, n).clip(0, 1)
    variance = rng.normal(32, 8, n).clip(0, 50)
    X.append(np.stack([brightness, saturation, dark_ratio, brown_ratio, variance], axis=1))
    y.append(np.full(n, 2))

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y


def main():
    print("Membuat dataset sintetis...")
    X, y = generate_synthetic_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("Menormalisasi fitur (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Melatih model Logistic Regression (multinomial)...")
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAkurasi pada data uji: {acc*100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Simpan model sebagai JSON (bobot, bias, parameter scaler)
    export = {
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": model.coef_.tolist(),        # shape (3, 5)
        "intercept": model.intercept_.tolist(),  # shape (3,)
        "accuracy": acc,
    }

    with open("fruit_model.json", "w") as f:
        json.dump(export, f, indent=2)

    with open("akurasi.txt", "w") as f:
        f.write(f"Akurasi model: {acc*100:.2f}%\n")
        f.write(f"Jumlah data latih: {len(X_train)}\n")
        f.write(f"Jumlah data uji: {len(X_test)}\n")
        f.write("Algoritma: Logistic Regression (multinomial / softmax)\n")

    print("Model disimpan sebagai fruit_model.json")
    print("Ringkasan akurasi disimpan sebagai akurasi.txt")


if __name__ == "__main__":
    main()
