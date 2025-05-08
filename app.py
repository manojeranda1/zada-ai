from flask import Flask, request, jsonify, send_file, render_template
import io
import os
from PIL import Image, ImageFilter, ImageEnhance  # Add ImageFilter and ImageEnhance to imports
import rembg
import logging
from werkzeug.utils import secure_filename
from flask_cors import CORS  # Import CORS

# Create Flask app with static folder configuration
app = Flask(__name__, static_url_path='', static_folder='static')

# Enable CORS for the app
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
OUTPUT_SIZE = (2000, 2000)
BACKGROUND_COLOR = "#f7f7f7"  # Light gray background
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB limit

# Initialize session for rembg (loaded once at startup)
session = None

first_request_processed = False

@app.before_request
def before_first_request_func():
    global first_request_processed
    if not first_request_processed:
        # Your code that you want to run only once
        initialize_session()  # Initialize the AI model session
        first_request_processed = True

# The rest of your code follows
def initialize_session():
    global session
    session = rembg.new_session()  # Consider 'u2net_human_seg' for human images
    logger.info("AI model session initialized")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def enhance_lighting(image):
    """Adjust brightness, contrast, color, and reduce noise for professional touch-up."""
    # Adjust brightness
    brightness_enhancer = ImageEnhance.Brightness(image)
    image = brightness_enhancer.enhance(1.1)  # Slightly brighter

    # Adjust contrast
    contrast_enhancer = ImageEnhance.Contrast(image)
    image = contrast_enhancer.enhance(1)  # Boost contrast

    # Adjust color saturation
    color_enhancer = ImageEnhance.Color(image)
    image = color_enhancer.enhance(1.0)  # Boost color saturation

    return image

def sharpen_edges_softly(image):
    """Apply targeted sharpening for crisp, natural edges."""
    return image.filter(ImageFilter.UnsharpMask(
        radius=0.1,  # Smaller radius for precision
        percent=100,  # Reduced strength for natural look
        threshold=4   # Higher threshold to avoid noise
    ))

def process_image(image_data):
    try:
        # Open and validate image
        input_image = Image.open(io.BytesIO(image_data)).convert('RGBA')
        
        # Remove background with preloaded session
        logger.info("Processing image...")
        output = rembg.remove(
            input_image,
            session=session,
            alpha_matting=False  # Disabled for sharp edges
        )
        
        # Convert to RGBA to ensure alpha channel is preserved
        output = output.convert('RGBA')
        
        # Calculate scaling parameters
        bg_color_rgb = hex_to_rgb(BACKGROUND_COLOR)
        output_ratio = min(OUTPUT_SIZE[0] / output.width, 
                          OUTPUT_SIZE[1] / output.height)
        new_size = (int(output.width * output_ratio), 
                   int(output.height * output_ratio))
        
        # Select resampling method
        resample = Image.LANCZOS
        if output_ratio > 1:  # Upscaling
            resample = Image.BICUBIC
        
        # Resize directly with optimal resampling
        resized_output = output.resize(new_size, resample=resample)
        
        # Enhance lighting for professional touch-up
        resized_output = enhance_lighting(resized_output)
        
        # Sharpen edges softly for crisp, natural look
        resized_output = sharpen_edges_softly(resized_output)
        
        # Create final image with centered position
        final_image = Image.new('RGBA', OUTPUT_SIZE, (0, 0, 0, 0))  # Transparent background
        position = (
            (OUTPUT_SIZE[0] - new_size[0]) // 2,
            (OUTPUT_SIZE[1] - new_size[1]) // 2
        )
        
        # Paste with alpha mask for sharp composition
        final_image.paste(resized_output, position, resized_output)
        
        # Convert to RGB with a solid background color
        final_image_rgb = Image.new('RGB', OUTPUT_SIZE, bg_color_rgb)
        final_image_rgb.paste(final_image, mask=final_image.split()[3])  # Use alpha channel as mask
        
        # Optimized output with efficient compression
        output_bytes = io.BytesIO()
        final_image_rgb.save(
            output_bytes,
            format='PNG',
            optimize=True,  # Enable optimization
            compress_level=1  # Lower compression level for smaller size without quality loss
        )
        output_bytes.seek(0)
        return output_bytes
    
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise

@app.route('/api/remove-background', methods=['POST'])
def remove_background():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    try:
        processed_image = process_image(file.read())
        return send_file(
            processed_image,
            mimetype='image/png',
            as_attachment=True,
            download_name=f"bg_removed_{secure_filename(file.filename)}.png"
        )
    
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({"error": "Image processing failed"}), 500

@app.route('/health')
def health_check():
    return jsonify({"status": "ok", "model": "active"}), 200

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)