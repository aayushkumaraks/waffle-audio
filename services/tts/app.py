from .src.tts_service import TTSService

tts = TTSService()

tts.start()

print("TTS ready")

input("Press Enter to exit...")

tts.stop()