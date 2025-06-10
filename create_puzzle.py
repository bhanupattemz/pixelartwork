import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from skimage import feature
import matplotlib.pyplot as plt
import os
import copy
import base64
from io import BytesIO
from PIL import Image
import numpy as np



def load_and_preprocess(image_path):
    """Load and prepare the image for processing with 3:4 aspect ratio cropping"""
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not load image from path: {image_path}")
        return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width = img.shape[:2]
    target_ratio = 3/4

    if width/height > target_ratio:
        new_width = int(height * target_ratio)
        start_x = (width - new_width) // 2
        img_cropped = img[:, start_x:start_x+new_width]
    else:
        new_height = int(width / target_ratio)
        start_y = (height - new_height) // 2
        img_cropped = img[start_y:start_y+new_height, :]

    img_resized = cv2.resize(img_cropped, (600, 800), interpolation=cv2.INTER_AREA)
    img_lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    return img_resized, img_lab

def detect_facial_features(img):
   
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = feature.canny(gray, sigma=2)

    feature_map = np.zeros_like(gray, dtype=np.float32)
    feature_map[edges] = 1.0

    h, w = img.shape[:2]
    center_y, center_x = h // 2, w // 2
    face_radius = min(h, w) // 3

    y_coords, x_coords = np.ogrid[:h, :w]
    mask = ((y_coords - center_y)**2 + (x_coords - center_x)**2) <= face_radius**2
    feature_map[mask] += 0.5

    feature_map = np.clip(feature_map, 0, 1)
    feature_map = cv2.GaussianBlur(feature_map, (15, 15), 0)
    return feature_map

def floyd_steinberg_dithering(img, feature_map, palette_size=5):
    if img is None or feature_map is None:
        return None
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab_img.shape[:2]
    output = lab_img.copy()

    pixels = lab_img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=palette_size, random_state=42, n_init=10)
    kmeans.fit(pixels)
    palette = kmeans.cluster_centers_

    for y in range(h-1):
        for x in range(w-1):
            old_pixel = output[y, x].copy()
            closest_idx = np.argmin(np.sum((palette - old_pixel)**2, axis=1))
            new_pixel = palette[closest_idx]
            output[y, x] = new_pixel

            quant_error = old_pixel - new_pixel
            feature_weight = feature_map[y, x]
            error_diffusion = 1.0 - 0.5 * feature_weight

            if x + 1 < w:
                output[y, x+1] += quant_error * 0.4375 * error_diffusion
            if y + 1 < h:
                if x > 0:
                    output[y+1, x-1] += quant_error * 0.1875 * error_diffusion
                output[y+1, x] += quant_error * 0.3125 * error_diffusion
                if x + 1 < w:
                    output[y+1, x+1] += quant_error * 0.0625 * error_diffusion

    output = np.clip(output, 0, 255).astype(np.uint8)
    return cv2.cvtColor(output, cv2.COLOR_LAB2RGB)

def create_puzzle_grid(dithered_img):
    if dithered_img is None:
        return None, None
    PIXELATED_WIDTH, PIXELATED_HEIGHT = 30, 40
    grid_canvas = np.ones((PIXELATED_HEIGHT, PIXELATED_WIDTH, 3), dtype=np.uint8) * 255

    h, w = dithered_img.shape[:2]
    original_aspect = w / h
    grid_aspect = PIXELATED_WIDTH / PIXELATED_HEIGHT

    if original_aspect > grid_aspect:
        new_width = PIXELATED_WIDTH
        new_height = int(new_width / original_aspect)
    else:
        new_height = PIXELATED_HEIGHT
        new_width = int(new_height * original_aspect)

    resized_img = cv2.resize(dithered_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    y_offset = (PIXELATED_HEIGHT - new_height) // 2
    x_offset = (PIXELATED_WIDTH - new_width) // 2
    grid_canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_img

    grid_gray = cv2.cvtColor(grid_canvas, cv2.COLOR_RGB2GRAY)
    grid_lab = cv2.cvtColor(grid_canvas, cv2.COLOR_RGB2LAB)

    return grid_gray, grid_lab

def generate_color_assignments(grid_lab):
    """Generate color assignments for 300 puzzle pieces"""
    # Added a check to prevent error if grid_lab is None
    if grid_lab is None:
        return []
    COLORS = [
        {"name": "red", "rgb": (255, 51, 51), "text": (255, 255, 255)},
        {"name": "blue", "rgb": (51, 102, 204), "text": (255, 255, 255)},
        {"name": "green", "rgb": (51, 204, 51), "text": (255, 255, 255)},
        {"name": "orange", "rgb": (255, 153, 0), "text": (255, 255, 255)},
        {"name": "purple", "rgb": (153, 51, 204), "text": (255, 255, 255)}
    ]

    h, w, _ = grid_lab.shape
    blocks_h, blocks_w = h // 2, w // 2
    block_info = []

    for y in range(blocks_h):
        for x in range(blocks_w):
            block = grid_lab[2*y:2*y+2, 2*x:2*x+2]
            avg_color = np.mean(block, axis=(0, 1))
            block_info.append({
                "x": x, "y": y,
                "brightness": avg_color[0]/255.0,
                "avg_color": avg_color
            })

    block_info.sort(key=lambda p: p["brightness"])
    blocks_per_color = len(block_info) // len(COLORS)
    color_counters = {color["name"]: 1 for color in COLORS}

    for i, block in enumerate(block_info):
        color_index = i // blocks_per_color
        if color_index >= len(COLORS):
            color_index = len(COLORS) - 1

        color = COLORS[color_index]
        block.update({
            "color": color["name"],
            "rgb": color["rgb"],
            "text_color": color["text"],
            "number": color_counters[color["name"]],
            "rotation": ((block["x"] + block["y"] + int(block["brightness"] * 10)) % 4) * 90
        })
        color_counters[color["name"]] += 1

    block_info.sort(key=lambda p: (p["y"], p["x"]))
    return block_info

def create_puzzle_key(block_info, original_img):
    """Create puzzle key with rotation-correct dot markers for 6/9"""
    # Added a check to prevent error if original_img is None or block_info is empty
    if original_img is None or not block_info:
        print("Warning: Cannot create puzzle key due to missing image or block info.")
        return Image.new('RGB', (100, 100), 'white') # Return a dummy image

    PUZZLE_WIDTH, PUZZLE_HEIGHT = 15, 20
    CELL_SIZE = 40
    PADDING_TOP = 80
    PADDING_LEFT = 50
    PADDING_BOTTOM = 30
    PADDING_RIGHT = 50

    key_width = PUZZLE_WIDTH * CELL_SIZE + PADDING_LEFT + PADDING_RIGHT
    key_height = PUZZLE_HEIGHT * CELL_SIZE + PADDING_TOP + PADDING_BOTTOM
    key_img = Image.new('RGB', (key_width, key_height), 'white')
    draw = ImageDraw.Draw(key_img)

    # Font loading
    font_loaded = False
    colab_fonts = [
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]

    for font_path in colab_fonts:
        try:
            font_large = ImageFont.truetype(font_path, 20)
            font_small = ImageFont.truetype(font_path, 12)
            font_cell = ImageFont.truetype(font_path, 20)
            font_loaded = True
            break
        except Exception:
            continue

    if not font_loaded:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_cell = ImageFont.load_default()

    # Add header text
    draw.text((key_width // 2, 20), "PUZZLE KEY",
              fill="black", font=font_large, anchor="mm")

    # Add small thumbnail
    thumb_size = 70
    thumb_img = Image.fromarray(original_img)
    thumb_img = thumb_img.resize((thumb_size, thumb_size), Image.LANCZOS)
    key_img.paste(thumb_img, (key_width - thumb_size - 10, 10))

    # Draw grid with colored cells
    for y in range(PUZZLE_HEIGHT):
        for x in range(PUZZLE_WIDTH):
            block = block_info[y * PUZZLE_WIDTH + x]
            cell_x = PADDING_LEFT + x * CELL_SIZE
            cell_y = PADDING_TOP + y * CELL_SIZE
            draw.rectangle(
                [(cell_x, cell_y), (cell_x + CELL_SIZE, cell_y + CELL_SIZE)],
                fill=block["rgb"]
            )

    # Draw grid lines
    for y in range(PUZZLE_HEIGHT + 1):
        y_pos = PADDING_TOP + y * CELL_SIZE
        draw.line([(PADDING_LEFT, y_pos),
                  (PADDING_LEFT + PUZZLE_WIDTH * CELL_SIZE, y_pos)],
                 fill="black", width=1)
    for x in range(PUZZLE_WIDTH + 1):
        x_pos = PADDING_LEFT + x * CELL_SIZE
        draw.line([(x_pos, PADDING_TOP),
                  (x_pos, PADDING_TOP + PUZZLE_HEIGHT * CELL_SIZE)],
                 fill="black", width=1)

    # Draw text numbers with proper sizing and rotation
    for y in range(PUZZLE_HEIGHT):
        for x in range(PUZZLE_WIDTH):
            block = block_info[y * PUZZLE_WIDTH + x]
            cell_x = PADDING_LEFT + x * CELL_SIZE
            cell_y = PADDING_TOP + y * CELL_SIZE

            big_size = CELL_SIZE * 3
            text_img = Image.new("RGBA", (big_size, big_size), (0,0,0,0))
            text_draw = ImageDraw.Draw(text_img)

            # Draw the number
            text_content = str(block['number'])
            text_draw.text(
                (big_size // 2, big_size // 2),
                text_content,
                fill=(255, 255, 255),
                font=font_cell,
                anchor="mm"
            )

            # Add dot for 6/9 (always at bottom before rotation)
            if block['number'] in [6, 9]:
                dot_size = 5
                dot_x = big_size // 2
                dot_y = big_size // 2 + 15
                text_draw.ellipse(
                    [(dot_x - dot_size//2, dot_y - dot_size//2),
                     (dot_x + dot_size//2, dot_y + dot_size//2)],
                    fill=(255, 255, 255)
                )

            # Rotate text
            rotated_text = text_img.rotate(
                -block["rotation"],
                center=(big_size // 2, big_size // 2),
                resample=Image.BICUBIC
            )

            offset_x = cell_x - (big_size - CELL_SIZE) // 2
            offset_y = cell_y - (big_size - CELL_SIZE) // 2
            key_img.paste(rotated_text, (offset_x, offset_y), rotated_text)

    # Draw thicker section borders
    for x in range(1, 3):
        x_pos = PADDING_LEFT + x * 5 * CELL_SIZE
        draw.line([(x_pos, PADDING_TOP),
                  (x_pos, PADDING_TOP + PUZZLE_HEIGHT * CELL_SIZE)],
                 fill="black", width=2)
    for y in range(1, 2):
        y_pos = PADDING_TOP + y * 10 * CELL_SIZE
        draw.line([(PADDING_LEFT, y_pos),
                  (PADDING_LEFT + PUZZLE_WIDTH * CELL_SIZE, y_pos)],
                 fill="black", width=2)

    key_img.info['dpi'] = (300, 300)
    return key_img

def create_print_compensated_puzzle_key(block_info, original_img, output_dir="output"):
    if original_img is None or not block_info:
        print("Warning: Cannot create print-compensated puzzle key due to missing image or block info.")
        return Image.new('RGB', (100, 100), 'white'), []

    compensated_block_info = copy.deepcopy(block_info)
    PUZZLE_WIDTH, PUZZLE_HEIGHT = 15, 20

    for block in compensated_block_info:
        if block["x"] == 7:
            continue
        block["x"] = (PUZZLE_WIDTH - 1) - block["x"]
        current_rotation = block["rotation"]
        if current_rotation == 90:
            block["rotation"] = 270
        elif current_rotation == 270:
            block["rotation"] = 90

    compensated_block_info.sort(key=lambda p: (p["y"], p["x"]))

    CELL_SIZE = 40
    PADDING_TOP = 80
    PADDING_LEFT = 50
    PADDING_BOTTOM = 30
    PADDING_RIGHT = 50

    key_width = PUZZLE_WIDTH * CELL_SIZE + PADDING_LEFT + PADDING_RIGHT
    key_height = PUZZLE_HEIGHT * CELL_SIZE + PADDING_TOP + PADDING_BOTTOM
    key_img = Image.new('RGB', (key_width, key_height), 'white')
    draw = ImageDraw.Draw(key_img)

    font_loaded = False
    colab_fonts = [
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]

    for font_path in colab_fonts:
        try:
            font_large = ImageFont.truetype(font_path, 20)
            font_small = ImageFont.truetype(font_path, 12)
            font_cell = ImageFont.truetype(font_path, 20)
            font_loaded = True
            break
        except Exception:
            continue

    if not font_loaded:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_cell = ImageFont.load_default()

    draw.text((key_width // 2, 20), "PRINT-COMPENSATED PUZZLE KEY",
              fill="red", font=font_large, anchor="mm")
    draw.text((key_width // 2, 45), "(For Double-Sided Printing)",
              fill="red", font=font_small, anchor="mm")

    thumb_size = 70
    thumb_img = Image.fromarray(original_img)
    thumb_img = thumb_img.resize((thumb_size, thumb_size), Image.LANCZOS)
    key_img.paste(thumb_img, (key_width - thumb_size - 10, 10))
    for y in range(PUZZLE_HEIGHT):
        for x in range(PUZZLE_WIDTH):
            block = compensated_block_info[y * PUZZLE_WIDTH + x]
            cell_x = PADDING_LEFT + x * CELL_SIZE
            cell_y = PADDING_TOP + y * CELL_SIZE
            draw.rectangle(
                [(cell_x, cell_y), (cell_x + CELL_SIZE, cell_y + CELL_SIZE)],
                fill=block["rgb"]
            )

    for y in range(PUZZLE_HEIGHT + 1):
        y_pos = PADDING_TOP + y * CELL_SIZE
        draw.line([(PADDING_LEFT, y_pos),
                  (PADDING_LEFT + PUZZLE_WIDTH * CELL_SIZE, y_pos)],
                 fill="black", width=1)
    for x in range(PUZZLE_WIDTH + 1):
        x_pos = PADDING_LEFT + x * CELL_SIZE
        draw.line([(x_pos, PADDING_TOP),
                  (x_pos, PADDING_TOP + PUZZLE_HEIGHT * CELL_SIZE)],
                 fill="black", width=2)

    for y in range(PUZZLE_HEIGHT):
        for x in range(PUZZLE_WIDTH):
            block = compensated_block_info[y * PUZZLE_WIDTH + x]
            cell_x = PADDING_LEFT + x * CELL_SIZE
            cell_y = PADDING_TOP + y * CELL_SIZE

            big_size = CELL_SIZE * 3
            text_img = Image.new("RGBA", (big_size, big_size), (0,0,0,0))
            text_draw = ImageDraw.Draw(text_img)

            text_content = str(block['number'])
            text_draw.text(
                (big_size // 2, big_size // 2),
                text_content,
                fill=(255, 255, 255),
                font=font_cell,
                anchor="mm"
            )

            if block['number'] in [6, 9]:
                dot_size = 5
                dot_x = big_size // 2
                dot_y = big_size // 2 + 15
                text_draw.ellipse(
                    [(dot_x - dot_size//2, dot_y - dot_size//2),
                     (dot_x + dot_size//2, dot_y + dot_size//2)],
                    fill=(255, 255, 255)
                )

            rotated_text = text_img.rotate(
                -block["rotation"],
                center=(big_size // 2, big_size // 2),
                resample=Image.BICUBIC
            )

            offset_x = cell_x - (big_size - CELL_SIZE) // 2
            offset_y = cell_y - (big_size - CELL_SIZE) // 2
            key_img.paste(rotated_text, (offset_x, offset_y), rotated_text)

    for x in range(1, 3):
        x_pos = PADDING_LEFT + x * 5 * CELL_SIZE
        draw.line([(x_pos, PADDING_TOP),
                  (x_pos, PADDING_TOP + PUZZLE_HEIGHT * CELL_SIZE)],
                 fill="black", width=2)
    for y in range(1, 2):
        y_pos = PADDING_TOP + y * 10 * CELL_SIZE
        draw.line([(PADDING_LEFT, y_pos),
                  (PADDING_LEFT + PUZZLE_WIDTH * CELL_SIZE, y_pos)],
                 fill="black", width=2)

    os.makedirs(output_dir, exist_ok=True)
    
    key_img.info['dpi'] = (300, 300)

    print(f"Print-compensated puzzle key saved with 8th column unchanged")
    return key_img, compensated_block_info


def images_to_data_urls(pixelated_img, puzzle_key, key_img):
    def image_to_data_url(img, format='PNG'):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if not isinstance(img, Image.Image):
            raise ValueError("Image must be PIL Image or numpy array")
        buffer = BytesIO()
        img.save(buffer, format=format)
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        mime_type = f'image/{format.lower()}'
        data_url = f'data:{mime_type};base64,{img_base64}'
        
        return data_url
    
    try:
        data_urls = {
            'pixelated_img': image_to_data_url(pixelated_img),
            'puzzle_key': image_to_data_url(puzzle_key),
            'key_img': image_to_data_url(key_img)
        }
        
        print("Successfully converted all images to data URLs")
        return data_urls
        
    except Exception as e:
        print(f"Error converting images to data URLs: {str(e)}")
        return None

def single_image_to_data_url(img, format='PNG'):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    if not isinstance(img, Image.Image):
        raise ValueError("Image must be PIL Image or numpy array")
    
    buffer = BytesIO()
    
    img.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    mime_type = f'image/{format.lower()}'
    data_url = f'data:{mime_type};base64,{img_base64}'
    
    return data_url
def create_puzzle_face(image_path, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    original_img, img_lab = load_and_preprocess(image_path)
    if original_img is None:
        print("Image loading failed. Aborting puzzle creation.")
        return None, None, None, [] 

    print("Detecting facial features...")
    feature_map = detect_facial_features(original_img)
    if feature_map is None: 
        print("Feature detection failed. Aborting.")
        return None, None, None, []

    print("Applying advanced dithering...")
    dithered_img = floyd_steinberg_dithering(original_img, feature_map)
    if dithered_img is None: 
        print("Dithering failed. Aborting.")
        return None, None, None, []

    print("Creating 1200-pixel puzzle grid...")
    grid_gray, grid_lab = create_puzzle_grid(dithered_img)
    if grid_gray is None or grid_lab is None:
        print("Grid creation failed. Aborting.")
        return None, None, None, []

    print("Generating color assignments for 300 pieces...")
    block_info = generate_color_assignments(grid_lab)
    if not block_info: 
        print("Color assignment failed. Aborting.")
        return None, None, None, []

    print("Creating 300-piece puzzle key...")
    puzzle_key = create_puzzle_key(block_info, original_img)

    print("Creating print-compensated puzzle key...")
    compensated_key, compensated_block_info = create_print_compensated_puzzle_key(
        block_info, original_img, output_dir
    ) 
    key_img=compensated_key
    if isinstance(puzzle_key, Image.Image) and puzzle_key.size != (100, 100):
        pixelated_img = Image.fromarray(cv2.resize(grid_gray, (300, 400), interpolation=cv2.INTER_NEAREST))
        data= images_to_data_urls(pixelated_img,puzzle_key,key_img)
        return data
    else:
        print("Puzzle creation failed due to missing image or data.")
        return None
