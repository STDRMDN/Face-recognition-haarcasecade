import cv2
from telegram import Bot
from datetime import datetime, timedelta

# Token bot Telegram
bot_token = '6781650545:AAEcfewn-UkGalciw_X1oCv3mdRHHOAmbAk'
chat_id = '5690490517'

# Inisialisasi bot
bot = Bot(token=bot_token)

# Variabel untuk menyimpan waktu terakhir pengiriman notifikasi
last_notification_time = None
notification_interval = timedelta(seconds=30)  # Jeda notifikasi 30 detik

# Fungsi untuk mengirim notifikasi
def send_notification(image, message):
    global last_notification_time
    now = datetime.now()
    if last_notification_time is None or now - last_notification_time >= notification_interval:
        # Encode gambar ke format JPEG
        success, encoded_image = cv2.imencode('.jpg', image)
        if success:
            # Kirim pesan dan gambar
            bot.send_message(chat_id=chat_id, text=message)
            bot.send_photo(chat_id=chat_id, photo=encoded_image.tobytes())
            last_notification_time = now
        else:
            print("Gagal meng-encode gambar.")

# Fungsi utama
def main():
    # Inisialisasi kamera
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal mengambil frame")
            break

        # Konversi ke grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Deteksi wajah
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Potong wajah dari frame
            face = frame[y:y + h, x:x + w]
            # Kirim notifikasi
            send_notification(face, "Wajah terdeteksi!")

        # Tampilkan frame
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Lepaskan kamera dan tutup jendela
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
