

document.getElementById("converseForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = document.getElementById("message").value;
    console.log("Message: ", message);
    const res = await fetch("/api/converse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message })
    });

    const data = await res.json();
    console.log("Response:", data);
});