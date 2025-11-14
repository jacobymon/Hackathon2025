// static/requesthandler.js
console.log("✅ JS loaded successfully");
import { convertWavToMp3 } from './audio_recording.js';
document.addEventListener("DOMContentLoaded", () => {
  // Handle TTS
  const ttsForm = document.getElementById("ttsForm");
  if (ttsForm) {
    console.log("ttsForm found");
    ttsForm.addEventListener("submit", async (e) => {
      e.preventDefault();

      const text = document.getElementById("ttsText").value;
      console.log("Sending TTS:", text);

      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });

      const data = await res.json();
      console.log("Response from Flask (TTS):", data);
    });
  }

  // ===============================
  // 🎙️ STT RECORDING FEATURE
  // ===============================
  const recordBtn = document.getElementById("recordBtn");
  const uploadBtn = document.getElementById("uploadBtn");
  const playback = document.getElementById("playback");
  

  let mediaRecorder;
  let audioChunks = [];
  let audioBlob;

  if (recordBtn && uploadBtn) {
    recordBtn.addEventListener("click", async () => {
      if (!mediaRecorder || mediaRecorder.state === "inactive") {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
          audioBlob = new Blob(audioChunks, { type: "audio/webm" });
          playback.src = URL.createObjectURL(audioBlob);
          uploadBtn.disabled = false;
        };

        mediaRecorder.start();
        recordBtn.textContent = "⏹ Stop Recording";
        uploadBtn.disabled = true;
        sttResult.textContent = "Recording...";
      } else {
        mediaRecorder.stop();
        recordBtn.textContent = "🎙 Start Recording";
        sttResult.textContent = "Recording stopped.";
      }
    });
    uploadBtn.addEventListener("click", async () => {
      if (!audioBlob) return alert("Please record something first.");
      console.log("AUDIO BLOB: ", audioBlob);
      const formData = new FormData();
      formData.append("audio", audioBlob, "recording.webm");
      console.log("FORM DATA WITH AUDIO FILE: ",formData);
      try {
        console.log("SENDING AUDIO FILE TO STT");
        const res = await fetch("/api/stt", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();
        sttResult.textContent = data.text || "No text recognized.";
      } catch (err) {
        console.error("❌ Upload failed:", err);
      }
    });
  }
});

