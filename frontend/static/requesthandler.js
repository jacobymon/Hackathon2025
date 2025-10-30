// static/requesthandler.js
console.log("✅ JS loaded successfully");
document.addEventListener("DOMContentLoaded", () => {
  // Handle TTS
  const ttsForm = document.getElementById("ttsForm");
  if (ttsForm) {
    console.log("ttsForm found");
    ttsForm.addEventListener("submit", async (e) => {
      e.preventDefault();

      const text = document.getElementById("ttsText").value;
      console.log("Sending TTS:", text);

      const res = await fetch("/api/converse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });

      const data = await res.json();
      console.log("Response from Flask (TTS):", data);
    });
  }
});