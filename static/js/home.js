function handleFileSelect(input) {
    const file = input.files[0];
    const fileInfo = document.getElementById('fileInfo');
    const fileDetails = document.getElementById('fileDetails');
    const submitBtn = document.getElementById('fileSubmitBtn');

    if (file) {
        fileDetails.innerHTML = `
                    <strong>${file.name}</strong><br>
                    <small>${(file.size / 1024 / 1024).toFixed(2)} MB</small>
                `;
        fileInfo.style.display = 'block';
        submitBtn.disabled = false;
    } else {
        fileInfo.style.display = 'none';
        submitBtn.disabled = true;
    }
}

const fileUploadArea = document.querySelector('.home-file-upload-area');

fileUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileUploadArea.classList.add('dragover');
});

fileUploadArea.addEventListener('dragleave', () => {
    fileUploadArea.classList.remove('dragover');
});

fileUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    fileUploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        const dt = new DataTransfer();
        dt.items.add(files[0]);
        document.getElementById('fileInput').files = dt.files;
        handleFileSelect(document.getElementById('fileInput'));
    }
});

document.getElementById('fileUploadForm').addEventListener('submit', (e) => {
    const fileInput = document.getElementById('fileInput');
    if (fileInput.files.length > 0) {
        document.getElementById('loadingDiv').style.display = 'block';
    } else {
        e.preventDefault();
    }
});

