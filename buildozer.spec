[app]
title = Sortir - Kualitas Buah
package.name = sortirbuah
package.domain = org.contohapp

source.dir = .
source.include_exts = py,kv,json,png,jpg,jpeg,txt

version = 1.0

# Dependensi Python yang dibutuhkan aplikasi (harus tersedia di python-for-android)
requirements = python3,kivy==2.3.0,pillow,numpy,plyer

# Orientasi aplikasi
orientation = portrait
fullscreen = 0

# Ikon & splash (opsional, tambahkan file sendiri lalu aktifkan baris di bawah)
# icon.filename = %(source.dir)s/assets/icon.png
# presplash.filename = %(source.dir)s/assets/presplash.png

[buildozer]
log_level = 2
warn_on_root = 1

[app:android]
android.accept_sdk_license = True

# Izin yang dibutuhkan: kamera + akses penyimpanan (untuk ambil/pilih foto)
android.permissions = CAMERA,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE

# Target & minimum API level Android
android.api = 33
android.minapi = 21
android.ndk = 25b
android.archs = arm64-v8a, armeabi-v7a

android.allow_backup = True
