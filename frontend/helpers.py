import sounddevice as sd
from pydub import AudioSegment

def record_audio(duration=5, samplerate=44100, channels=1, mp3_filename="output.mp3"):
    print(f"🎙️ Recording for {duration} seconds...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("✅ Recording complete.")

    # Convert NumPy array to WAV-like bytes using pydub
    audio = AudioSegment(
        recording.tobytes(),
        frame_rate=samplerate,
        sample_width=recording.dtype.itemsize,
        channels=channels
    )

    # Export as MP3
    audio.export(mp3_filename, format="mp3")
    print(f"💾 Saved as {mp3_filename}")