import telegram
import asyncio

# Ganti dengan token bot Anda yang benar
bot_token = '6781650545:AAEcfewn-UkGalciw_X1oCv3mdRHHOAmbAk'
chat_id = '7362434981'  # Ganti dengan chat ID Anda

# Inisialisasi bot
bot = telegram.Bot(token=bot_token)

async def send_test_message():
    """Mengirim pesan uji coba ke Telegram."""
    try:
        await bot.send_message(chat_id=chat_id, text="Bot Face Recognition berhasil terhubung!")
        print("Pesan uji coba berhasil dikirim.")
    except Exception as e:
        print(f"Terjadi kesalahan saat mengirim pesan: {e}")

# Jalankan fungsi asynchronous
asyncio.run(send_test_message())
