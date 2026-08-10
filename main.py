"""
main.py
Aplikasi mobile (Kivy) untuk klasifikasi kualitas buah menggunakan model
Machine Learning (Logistic Regression) yang sudah dilatih di
model/train_model.py. Bisa di-build jadi APK Android lewat Buildozer.

Alur:
  1. User ambil foto (kamera) atau pilih dari galeri.
  2. utils.extract_features() mengekstrak fitur citra dari foto.
  3. utils.FruitClassifier melakukan inferensi model ML (softmax regression).
  4. Hasil (Grade + confidence + rincian fitur) ditampilkan di UI.
"""

import os
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty, BooleanProperty, ListProperty
from kivy.clock import Clock

from utils import FruitClassifier

PLACEHOLDER_IMAGE = ""

# ---- Permission Android (aman di-skip kalau bukan di Android) ----
try:
    from android.permissions import request_permissions, Permission
    ANDROID = True
except ImportError:
    ANDROID = False

try:
    from plyer import camera, filechooser
except Exception:
    camera = None
    filechooser = None


GRADE_COLORS = {
    "Segar": [0.184, 0.322, 0.2, 1],   # forest green
    "Sedang": [0.725, 0.482, 0.071, 1],  # gold
    "Busuk": [0.612, 0.165, 0.09, 1],   # rust red
}


class Root(BoxLayout):
    image_source = StringProperty(PLACEHOLDER_IMAGE)
    has_image = BooleanProperty(False)
    has_result = BooleanProperty(False)

    grade_text = StringProperty("")
    score_text = StringProperty("")
    probs_text = StringProperty("")
    note_text = StringProperty("")
    status_text = StringProperty("Ambil atau pilih foto buah untuk mulai")
    grade_color = ListProperty([0.184, 0.322, 0.2, 1])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classifier = FruitClassifier()
        self._captured_path = None
        if ANDROID:
            request_permissions([
                Permission.CAMERA,
                Permission.READ_EXTERNAL_STORAGE,
                Permission.WRITE_EXTERNAL_STORAGE,
            ])

    # ---------------- Ambil foto dari kamera ----------------
    def take_photo(self):
        if camera is None:
            self.status_text = "Kamera tidak tersedia di platform ini (coba di HP)."
            return
        save_path = os.path.join(App.get_running_app().user_data_dir, "capture.jpg")
        try:
            camera.take_picture(filename=save_path, on_complete=self._on_photo_taken)
            self.status_text = "Membuka kamera..."
        except NotImplementedError:
            self.status_text = "Kamera tidak didukung di platform ini."

    def _on_photo_taken(self, path):
        if path and os.path.exists(path):
            self._set_image(path)
        else:
            self.status_text = "Pengambilan foto dibatalkan."

    # ---------------- Pilih dari galeri ----------------
    def pick_from_gallery(self):
        if filechooser is None:
            self.status_text = "File chooser tidak tersedia di platform ini."
            return
        try:
            filechooser.open_file(
                on_selection=self._on_gallery_selected,
                filters=[["Gambar", "*.jpg", "*.jpeg", "*.png"]],
            )
        except Exception as e:
            self.status_text = f"Gagal membuka galeri: {e}"

    def _on_gallery_selected(self, selection):
        if selection:
            self._set_image(selection[0])
        else:
            self.status_text = "Tidak ada foto dipilih."

    # ---------------- Helper ----------------
    def _set_image(self, path):
        self._captured_path = path
        self.image_source = path
        self.has_image = True
        self.has_result = False
        self.status_text = "Foto siap. Tekan KLASIFIKASIKAN."

    # ---------------- Klasifikasi ----------------
    def classify_image(self):
        if not self._captured_path:
            return
        self.status_text = "Menganalisis foto..."
        Clock.schedule_once(lambda dt: self._run_classification(), 0.05)

    def _run_classification(self):
        try:
            label, confidence, probs, feats = self.classifier.predict_image(self._captured_path)
        except Exception as e:
            self.status_text = f"Gagal menganalisis: {e}"
            return

        note_map = {
            "Segar": "Warna cerah & merata, bercak minim. Layak jual kualitas premium.",
            "Sedang": "Mulai ada bercak / warna memudar. Sebaiknya segera dipakai atau dijual cepat.",
            "Busuk": "Bercak gelap & variasi warna tinggi. Tidak disarankan untuk dijual.",
        }

        self.grade_text = f"GRADE: {label.upper()}"
        self.grade_color = GRADE_COLORS.get(label, [0.2, 0.2, 0.2, 1])
        self.score_text = f"Tingkat keyakinan model: {confidence*100:.1f}%"

        probs_lines = "\n".join(
            f"{k}: {v*100:.1f}%" for k, v in sorted(probs.items(), key=lambda kv: -kv[1])
        )
        self.probs_text = "Probabilitas tiap kelas:\n" + probs_lines
        self.note_text = note_map.get(label, "")

        self.has_result = True
        self.status_text = "Klasifikasi selesai."


class FruitQualityApp(App):
    def build(self):
        self.title = "Sortir - Klasifikasi Kualitas Buah"
        Builder.load_file("fruit.kv")
        return Root()


if __name__ == "__main__":
    FruitQualityApp().run()