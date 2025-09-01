const filteredOutput = document.getElementById('filtered-output');
const startButton = document.getElementById('start-button');
const stopButton = document.getElementById('stop-button');
const statusText = document.getElementById('status-text');

let isStreaming = false;

// Fungsi untuk memulai streaming
startButton.addEventListener('click', () => {
    if (!isStreaming) {
        // Tampilkan video langsung dari server Flask
        filteredOutput.src = "/video_feed";
        statusText.textContent = 'Status: Terhubung ke server dan stream dimulai!';
        isStreaming = true;
    }
});

// Fungsi untuk menghentikan streaming
stopButton.addEventListener('click', () => {
    if (isStreaming) {
        // Menghentikan streaming dengan mengganti sumber gambar
        filteredOutput.src = "https://via.placeholder.com/640x480.png?text=Stream+Stopped";
        statusText.textContent = 'Status: Stream dihentikan.';
        isStreaming = false;
    }
});