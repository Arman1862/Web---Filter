const filteredOutput = document.getElementById('filtered-output');
const startButton = document.getElementById('start-button');
const stopButton = document.getElementById('stop-button');
const statusText = document.getElementById('status-text');

// Buat koneksi ke server WebSocket
const socket = io();

// Dengarkan event 'video_frame' dari server
socket.on('video_frame', function(data) {
    filteredOutput.src = 'data:image/jpeg;base64,' + data.image;
});

// Kirim event 'start_stream' ke server saat tombol diklik
startButton.addEventListener('click', () => {
    socket.emit('start_stream');
    statusText.textContent = 'Status: Stream dimulai via WebSocket!';
});

// Kirim event 'stop_stream' ke server saat tombol diklik
stopButton.addEventListener('click', () => {
    socket.emit('stop_stream');
    statusText.textContent = 'Status: Stream dihentikan.';
    filteredOutput.src = "https://via.placeholder.com/640x480.png?text=Stream+Stopped";
});