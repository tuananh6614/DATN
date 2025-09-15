# save as gtts_vi.py
from gtts import gTTS

def save_vi_mp3(text="cảm ơn quý khách", filename="xin_cam_on.mp3", lang="vi"):
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)
    print(f"Saved {filename}")

if __name__ == "__main__":
    save_vi_mp3("cảm ơn quý khách", "xin_cam_on.mp3")
