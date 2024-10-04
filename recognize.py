import cv2
import pickle
import telegram
import asyncio
from datetime import datetime, timedelta
import os

# Ganti dengan token bot Anda yang benar
bot_token = '6781650545:AAEcfewn-UkGalciw_X1oCv3mdRHHOAmbAk'
chat_id = '5690490517'

# Inisialisasi bot Telegram
bot = telegram.Bot(token=bot_token)

save_path = r"C:\xampp\htdocs\Face-Recognition-haarcasecade\unknown"

# Pastikan folder untuk menyimpan gambar ada
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Variabel global untuk menyimpan waktu terakhir pengiriman notifikasi
last_notification_time = None
notification_interval = timedelta(seconds=30)  # Atur jeda notifikasi (1 menit)

async def test_koneksi_telegram():
    """Menguji koneksi bot Telegram dengan mengirimkan pesan uji."""
    try:
        await bot.send_message(chat_id=chat_id, text="Pesan uji dari skrip face recognition.")
        print("Pesan uji berhasil dikirim.")
    except Exception as e:
        print(f"Gagal mengirim pesan uji: {e}")

async def kirim_notifikasi_telegram(full_frame, pesan):
    """Kirim notifikasi dan gambar full frame ke Telegram ketika wajah tidak dikenal terdeteksi."""
    try:
        # Encode gambar full frame langsung ke dalam format JPEG tanpa menyimpannya di disk
        success, encoded_image = cv2.imencode('.jpg', full_frame)
        if success:
            # Kirim pesan notifikasi
            await bot.send_message(chat_id=chat_id, text=pesan)
            print("Notifikasi berhasil dikirim.")

            # Kirim gambar full frame langsung dari memori
            await bot.send_photo(chat_id=chat_id, photo=encoded_image.tobytes())
            print(f"Gambar full frame berhasil dikirim.")
        else:
            print("Gagal meng-encode gambar full frame.")
    except Exception as e:
        print(f"Kesalahan saat mengirim notifikasi Telegram: {e}")

def muat_mapping_nama(nama_file):
    """Memuat mapping nama dari file."""
    with open(nama_file, 'rb') as f:
        return pickle.load(f)

async def gambar_batas(img, classifier, scaleFactor, minNeighbors, warna, clf, nama_ke_id):
    """Gambar batas di sekitar wajah yang terdeteksi dan identifikasi mereka."""
    global last_notification_time

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []

    unknown_faces_detected = False

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), warna, 2)
        id, confidence = clf.predict(gray_img[y:y + h, x:x + w])
        threshold = 50  # Ubah threshold sesuai kebutuhan
        if confidence > threshold:
            nama = "Unknown"
        else:
            nama = [k for k, v in nama_ke_id.items() if v == id]
            nama = nama[0] if nama else "Unknown"

        cv2.putText(img, f'{nama} ({int(confidence)})', (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, warna, 1, cv2.LINE_AA)

        # Jika wajah tidak dikenal, set flag unknown_faces_detected
        if nama == "Unknown":
            unknown_faces_detected = True

    # Kirim notifikasi setiap kali ada wajah tidak dikenal, hanya jika sudah melewati interval waktu yang diatur
    now = datetime.now()
    if unknown_faces_detected and (last_notification_time is None or now - last_notification_time >= notification_interval):
        await kirim_notifikasi_telegram(img, "Wajah tidak dikenal terdeteksi! (Full Frame)")
        last_notification_time = now  # Update waktu notifikasi terakhir
        
    return coords

async def mengenali(img, clf, faceCascade, nama_ke_id):
    """Mengenali wajah dalam gambar yang ditangkap."""
    warna = {"biru": (255, 0, 0), "merah": (0, 0, 255), "hijau": (0, 255, 0), "putih": (255, 255, 255)}
    coords = await gambar_batas(img, faceCascade, 1.1, 10, warna["putih"], clf, nama_ke_id)
    return img

async def main():
    # Uji koneksi bot Telegram
    await test_koneksi_telegram()

    # Muat classifier dan recognizer
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")
    nama_ke_id = muat_mapping_nama('name_to_id.pkl')

    # Mulai video capture
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, img = video_capture.read()
        if not ret:
            print("Gagal mengambil frame")
            break

        # Hapus variabel `notification_sent` agar notifikasi dikirim setiap kali ada wajah tidak dikenal
        img = await mengenali(img, clf, faceCascade, nama_ke_id)
        cv2.imshow("Deteksi Wajah", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Lepaskan video capture dan tutup semua window
    video_capture.release()
    cv2.destroyAllWindows()

# Jalankan fungsi main
asyncio.run(main())
