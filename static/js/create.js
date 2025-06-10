
function downloadImage(imageId, filename) {
    const loading = document.getElementById('create-loading');
    loading.style.display = 'block';

    setTimeout(() => {
        try {
            const img = document.getElementById(imageId);
            if (!img) {
                alert('Image not found!');
                loading.style.display = 'none';
                return;
            }
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            ctx.drawImage(img, 0, 0);
            canvas.toBlob(function (blob) {
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                loading.style.display = 'none';
            }, 'image/png', 1.0);

        } catch (error) {
            console.error('Download error:', error);
            alert('Download failed. Please try right-clicking the image and selecting "Save image as..."');
            loading.style.display = 'none';
        }
    }, 500);
}
document.addEventListener('DOMContentLoaded', function () {
    const images = document.querySelectorAll('.create-puzzle-image');
    images.forEach(img => {
        img.style.cursor = 'pointer';
        img.addEventListener('click', function () {
            const modal = document.createElement('div');
            modal.style.cssText = `
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        background: rgba(0,0,0,0.9);
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        z-index: 1000;
                        cursor: pointer;
                    `;

            const modalImg = document.createElement('img');
            modalImg.src = this.src;
            modalImg.style.cssText = `
                        max-width: 90%;
                        max-height: 90%;
                        border-radius: 10px;
                        box-shadow: 0 0 50px rgba(255,255,255,0.1);
                    `;

            modal.appendChild(modalImg);
            document.body.appendChild(modal);

            modal.addEventListener('click', function () {
                document.body.removeChild(modal);
            });
        });
    });
});
window.addEventListener('load', function () {
    const items = document.querySelectorAll('.puzzle-item');
    items.forEach((item, index) => {
        item.style.opacity = '0';
        item.style.transform = 'translateY(30px)';
        setTimeout(() => {
            item.style.transition = 'all 0.6s ease';
            item.style.opacity = '1';
            item.style.transform = 'translateY(0)';
        }, index * 200);
    });
});