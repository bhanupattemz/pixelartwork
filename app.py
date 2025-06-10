from flask import Flask, render_template, redirect, url_for, flash, request
from create_puzzle import create_puzzle_face
import os
import base64
from PIL import Image
import io
from werkzeug.utils import secure_filename
import traceback

app = Flask(__name__)
app.secret_key = 'your-secret-key-here' 

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/assets'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_base64_image(base64_data, filename):
    try:
        if base64_data.startswith('data:image'):
            base64_data = base64_data.split(',')[1]
        img_data = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(img_data))
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image.save(filepath)
        
        return filepath
    except Exception as e:
        print(f"Error saving base64 image: {e}")
        return None

@app.route('/', methods=["GET", "POST"])
def home():
    outputs = []
    if os.path.exists(OUTPUT_FOLDER):
        for item in os.listdir(OUTPUT_FOLDER):
            if item.endswith(('.png', '.jpg', '.jpeg')):
                outputs.append(item)
    
    return render_template("index.html", outputs=outputs)
@app.route('/create', methods=["POST"])
def create_puzzle():
    try:
        image_path = None
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                import time
                timestamp = str(int(time.time()))
                filename = f"{timestamp}_{filename}"
                image_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(image_path)
            else:
                flash("Please upload a valid image file (PNG, JPG, JPEG, GIF, BMP)", "error")
                return redirect(url_for("home"))
                
        elif 'raw-img' in request.form:
            base64_data = request.form.get("raw-img")
            if base64_data:
                import time
                timestamp = str(int(time.time()))
                filename = f"image_{timestamp}.png"
                image_path = save_base64_image(base64_data, filename)
                if not image_path:
                    flash("Failed to process image data", "error")
                    return redirect(url_for("home"))
            else:
                flash("No image data received", "error")
                return redirect(url_for("home"))
        else:
            flash("No image provided", "error")
            return redirect(url_for("home"))
        
        import time
        timestamp = str(int(time.time()))
        data = create_puzzle_face(image_path)
        
        if data:
            flash(f"Puzzle created successfully!", "success")
            return render_template("puzzle_view.html", data=data) 
        else:
            flash("Failed to create puzzle", "error")
            return redirect(url_for("home"))
            
    except Exception as e:
        print(f"Error creating puzzle: {e}")
        print(traceback.format_exc())
        flash(f"An error occurred while creating the puzzle: {str(e)}", "error")
        return redirect(url_for("home"))

if __name__ == "__main__":
    print("Starting Puzzle Face Creator Flask App...")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    app.run(debug=True, port=5002, host='0.0.0.0')