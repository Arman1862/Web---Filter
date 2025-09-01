document.addEventListener('DOMContentLoaded', (event) => {

    const videoInput = document.getElementById('webcam-input');
    const filteredOutput = document.getElementById('filtered-output');
    const startButton = document.getElementById('start-button');
    const stopButton = document.getElementById('stop-button');
    const statusText = document.getElementById('status-text');
    
    let isStreaming = false;
    let streamInterval;
    
    // Fungsi untuk memulai streaming
    startButton.addEventListener('click', () => {
        if (!isStreaming) {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    videoInput.srcObject = stream;
                    statusText.textContent = 'Status: Kamera lokal aktif.';
                    isStreaming = true;
    
                    streamInterval = setInterval(sendFrameToServer, 100); // Kirim frame setiap 100ms
                })
                .catch(error => {
                    console.error("Error accessing camera: ", error);
                    statusText.textContent = 'Status: Gagal mengakses kamera.';
                });
        }
    });
    
    // Fungsi untuk menghentikan streaming
    stopButton.addEventListener('click', () => {
        if (isStreaming) {
            clearInterval(streamInterval);
            filteredOutput.src = "https://via.placeholder.com/640x480.png?text=Stream+Stopped";
            statusText.textContent = 'Status: Stream dihentikan.';
            isStreaming = false;
            videoInput.srcObject = null;
        }
    });
    
    // Fungsi untuk mengirimkan frame ke server
    function sendFrameToServer() {
        if (videoInput.readyState === videoInput.HAVE_ENOUGH_DATA) {
            const canvas = document.createElement('canvas');
            canvas.width = videoInput.videoWidth;
            canvas.height = videoInput.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(videoInput, 0, 0, canvas.width, canvas.height);
    
            canvas.toBlob(blob => {
                fetch('/video_feed', {
                    method: 'POST',
                    body: blob
                })
                .then(response => response.blob())
                .then(blob => {
                    const imageUrl = URL.createObjectURL(blob);
                    filteredOutput.src = imageUrl;
                })
                .catch(error => console.error('Error:', error));
            }, 'image/jpeg');
        }
    }

});
