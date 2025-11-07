import os
import random
import math
import numpy as np
import cv2  # Needs opencv-python
from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageFilter

# --- New Imports for Dataset Generation ---
import csv
import argparse
from tqdm import tqdm


# -----------------------------------------------------------------------------
# ⚙️ FINAL SETTINGS (Tweak these variables for ultimate control)
# -----------------------------------------------------------------------------

# (All your existing settings remain unchanged)

# --- 1. Canvas Properties ---
AUTO_SCALE_BASE_HEIGHT = 115
# AUTO_SCALE_WIDTH_PER_CHAR = 53 # <-- This is now controlled by CHAR_SPACING_DEFAULT
BACKGROUND_TINT_COLOR = "#E0E0E0"
BACKGROUND_TINT_ALPHA = 30 # (0-255, 30 is very faint)
USE_FIXED_CANVAS_WIDTH = True
FIXED_CANVAS_WIDTH = 460
# --- 2. Text ---
CAPTCHA_TEXT = "FARTCOIN" # This is now the DEFAULT if no text is given
CAPTCHA_LENGTH = 8
CHAR_SET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
FONT_SIZE_MIN_PERCENT = 0.45
FONT_SIZE_MAX_PERCENT = 0.45
FONT_FILES = [
    'DejaVu Serif Book.ttf'
]
TEXT_COLORS = [
    '#095D05',  # Dark Green
    '#0A0579',
    '#0C0402'  
]

# --- 3. Text Distortion ---
WORD_WAVE_FREQUENCY = 190
WORD_WAVE_AMPLITUDE = 45
CHAR_PIXEL_FREQUENCY = 0.0
CHAR_PIXEL_AMPLITUDE = 5
CHAR_PIXEL_AMPLITUDE_VERTICAL = 15.0

FIRST_CHAR_HORIZONTAL_AMPLITUDE_OVERRIDE = 0
FIRST_CHAR_VERTICAL_AMPLITUDE_OVERRIDE =-20
FIRST_CHAR_VERTICAL_PINCH_MODE = "DOWN"
FIRST_CHAR_VERTICAL_OFFSET = 11
FIRST_CHAR_ROTATION_ANGLE = -15
FIRST_CHAR_SHEAR_ANGLE = 0


SECOND_CHAR_HORIZONTAL_AMPLITUDE_OVERRIDE = 0
SECOND_CHAR_VERTICAL_AMPLITUDE_OVERRIDE = 15.0
SECOND_CHAR_ROTATION_ANGLE = -15 # (e.g., 5)
SECOND_CHAR_VERTICAL_OFFSET = 10 # (e.g., 10 pixels down)
SECOND_CHAR_SHEAR_ANGLE = -15

ROTATION_ANGLE_MIN = -8
ROTATION_ANGLE_MAX = 8



SHEAR_ANGLE_MIN = -15
SHEAR_ANGLE_MAX = 15

CHAR_PIXEL_DENSITY = 0.90
JITTER_RANGE_PERCENT = 0.0
TEXT_START_X = 25
TEXT_START_Y = 28
VERTICAL_JITTER_RANGE_PIXELS = 0
CHAR_GAP_DEFAULT = 12
CHAR_GAP_1_TO_2_OVERRIDE = 4



# --- 5. Line Noise ---
NUM_LINES_MIN = 2
NUM_LINES_MAX = 3
LINE_WIDTH = 3
LINE_LENGTH_MIN = 50
LINE_LENGTH_MAX = 100
LINE_COLORS = ['#8B0000', '#05059E', '#808080', '#3B9D3B']
LINE_NOISE_ALPHA = 200



# --- 6. Cluster Dot Noise ---
BLOCK_SIZE = 1.5 # Size of each of the 4 quadrants in a cluster
TOTAL_DENSITY_PER_10K_PIXELS = 350
CLUSTER_TYPES = [
    {'color1': '#4E4E4E', 'color2': '#BBBBBB', 'density_weight': 150},
    {'color1': '#B5B5B5', 'color2': '#D1D1D1', 'density_weight': 150},
    {'color1': '#262626', 'color2': '#707070', 'density_weight': 150},
    {'color1': '#EAEAEA', 'color2': '#F0F0F0', 'density_weight': 150},
    {'color1': '#D59D9D', 'color2': '#D08F8F', 'density_weight': 40}, # Red
    {'color1': '#C98080', 'color2': '#EFD9D9', 'density_weight': 60}, # Red
    {'color1': '#C37272', 'color2': '#FAF3F3', 'density_weight': 60}, # Red
    {'color1': '#59AC59', 'color2': '#E1F0E1', 'density_weight': 60}, # Green
    {'color1': '#78BC78', 'color2': '#78BC78', 'density_weight': 40}, # Green
    {'color1': '#CDDCCD', 'color2': '#3A8E3A', 'density_weight': 60}, # Green
]

# --- 6B. NEW: Character-Only Foreground Noise ---
# This is drawn *on top* of the characters using the text mask
APPLY_CHAR_FOREGROUND_NOISE = False

APPLY_CHAR_CLUSTER_NOISE = False
CHAR_CLUSTER_DENSITY_PER_10K_PIXELS = 1 
CHAR_CLUSTER_BLOCK_SIZE = 2
CHAR_CLUSTER_NOISE_ALPHA = 120
# Add any cluster types you want to see ON your text here
CHAR_CLUSTER_TYPES = [
    {'color1': '#7B496E', 'color2': '#CCCCCC', 'density_weight': 5}
]



# --- 7. Final Image Effects ---
APPLY_GAUSSIAN_BLUR = True
GAUSSIAN_BLUR_RADIUS_MIN = 0.9
GAUSSIAN_BLUR_RADIUS_MAX = 0.9
FINAL_ROTATION_ANGLE = 0.0

APPLY_CHAR_BLUR = True
CHAR_BLUR_RADIUS_MIN = 0.3
CHAR_BLUR_RADIUS_MAX = 0.3

# -----------------------------------------------------------------------------
# 🆕 NEW: DATASET GENERATION SETTINGS
# -----------------------------------------------------------------------------
DEFAULT_TOTAL = 100  # Default number of images to generate
SPLIT_RATIO = (0.80, 0.10, 0.10)  # Train, validation, and test split
DEFAULT_OUT_ROOT = "captcha_dataset" # Will be created in the current folder
LABELS_CSV_NAME = "labels.csv"
SPLIT_NAMES = ["train", "val", "test"]

# --- Define the list of labels you want to generate ---
# You can expand this list with all your "tickers" or labels
DEFAULT_LABELS_LIST = [
    "BTC", "ETH", "DOGE", "SHIB", "PEPE", "WIF", "BONK", "FLOKI"
]


# =============================================================================
# 🖼️ IMAGE GENERATION FUNCTIONS (Unchanged)
# =============================================================================

def check_fonts():
    """Checks if font files exist and provides a helpful error."""
    if not FONT_FILES:
        print("❌ ERROR: The `FONT_FILES` list is empty. Please add font file paths.")
        return False
    for font_path in FONT_FILES:
        if not os.path.exists(font_path):
            print(f"❌ ERROR: Font file not found: '{font_path}'")
            print("➡️ Please download 'RobotoMono-VariableFont_wght.ttf' (or your chosen font) and place it in the same folder.")
            return False
    return True


# -----------------------------------------------------------------------------
# NEW FUNCTION: CHARACTER-ONLY CLUSTER NOISE
# -----------------------------------------------------------------------------
def draw_char_cluster_noise(image: Image.Image, text_mask: Image.Image):
    """
    Draws 2x2 block cluster noise *only* where the text mask is white.
    Uses CHAR_CLUSTER_ settings.
    """
    # print("Drawing character-only CLUSTER noise...") # Reduced print noise
    IMG_WIDTH, IMG_HEIGHT = image.size
    total_cluster_size = CHAR_CLUSTER_BLOCK_SIZE * 2
    
    # --- 1. Calculate Total Clusters to Draw ---
    total_pixels = IMG_WIDTH * IMG_HEIGHT
    # Calculate how many clusters to draw based on density
    total_clusters_to_draw = int((total_pixels / 10000.0) * CHAR_CLUSTER_DENSITY_PER_10K_PIXELS)
    
    # --- 2. Calculate Total Weight ---
    if not CHAR_CLUSTER_TYPES:
        # print("    Skipping: CHAR_CLUSTER_TYPES list is empty.")
        return image
        
    total_weight = sum(cluster_type['density_weight'] for cluster_type in CHAR_CLUSTER_TYPES)
    
    if total_weight == 0:
        # print("    WARNING: Total density_weight is 0. No clusters will be drawn.")
        return image

    # print(f"    Total character clusters to draw (randomly): {total_clusters_to_draw}")

    # --- 3. Create a temporary layer to draw noise on ---
    dot_layer = Image.new('RGBA', (IMG_WIDTH, IMG_HEIGHT), (0, 0, 0, 0))
    dot_drawer = ImageDraw.Draw(dot_layer)
    
    total_clusters_drawn = 0
    for cluster_type in CHAR_CLUSTER_TYPES:
        color1_rgba = ImageColor.getrgb(cluster_type['color1']) + (255,)
        color2_rgba = ImageColor.getrgb(cluster_type['color2']) + (255,)
        
        weight_proportion = cluster_type['density_weight'] / total_weight
        density_count = int(weight_proportion * total_clusters_to_draw)
        
        for _ in range(density_count):
            tlx = random.randint(0, IMG_WIDTH - total_cluster_size)
            tly = random.randint(0, IMG_HEIGHT - total_cluster_size)
            
            # Draw the 4 blocks of the cluster
            dot_drawer.rectangle(
                [tlx, tly, tlx + CHAR_CLUSTER_BLOCK_SIZE - 1, tly + CHAR_CLUSTER_BLOCK_SIZE - 1],
                fill=color1_rgba
            )
            dot_drawer.rectangle(
                [tlx + CHAR_CLUSTER_BLOCK_SIZE, tly, tlx + total_cluster_size - 1, tly + CHAR_CLUSTER_BLOCK_SIZE - 1],
                fill=color2_rgba
            )
            dot_drawer.rectangle(
                [tlx, tly + CHAR_CLUSTER_BLOCK_SIZE, tlx + CHAR_CLUSTER_BLOCK_SIZE - 1, tly + total_cluster_size - 1],
                fill=color2_rgba
            )
            dot_drawer.rectangle(
                [tlx + CHAR_CLUSTER_BLOCK_SIZE, tly + CHAR_CLUSTER_BLOCK_SIZE, tlx + total_cluster_size - 1, tly + total_cluster_size - 1],
                fill=color1_rgba
            )
        total_clusters_drawn += density_count

    # print(f"    Total clusters actually drawn: {total_clusters_drawn}")
    del dot_drawer
    
    # --- 4. Paste the noise layer ONTO the image, using the text_mask ---
    # This is the key: it only pastes the dots *where* the text mask is white.
    image.paste(dot_layer, (0, 0), text_mask)
    return image
def apply_wave_warp(pil_image: Image.Image, amplitude: float, frequency: float, vertical_amplitude: float, vertical_pinch_mode: str = "DEFAULT") -> Image.Image:
    
    cv_img = np.array(pil_image)
    rows, cols, _ = cv_img.shape
    
    map_x = np.zeros((rows, cols), dtype=np.float32)
    map_y = np.zeros((rows, cols), dtype=np.float32)

    v_amplitude = vertical_amplitude 
    h_amplitude = amplitude
    
    # --- Check for division by zero on frequency ---
    if frequency == 0:
        frequency = rows # Prevent error, effectively no horizontal wave
    
    for i in range(rows):
        # --- FIX 1: This is the HORIZONTAL (side-to-side) wave ---
        # It now correctly uses the 'frequency' parameter.
        v_progress = (i / frequency) * (2 * math.pi) 
        offset_x = int(h_amplitude * (-math.cos(v_progress)))

        for j in range(cols):
            # --- This is the VERTICAL "pinch" wave ---
            if vertical_pinch_mode == "DOWN":
                h_progress_down = (j / cols) * math.pi 
                offset_y = int(v_amplitude * (math.sin(h_progress_down))) 
            else: # "DEFAULT"
                h_progress_default = (j / cols) * (2 * math.pi)
                offset_y = int(v_amplitude * (-math.cos(h_progress_default)))
            
            new_x = j + offset_x
            new_y = i + offset_y
            
            # --- FIX 2: This is the CLAMPING logic for VERTICAL ---
            # This is the fix for your "jumpy" bug.
            
            if 0 <= new_y < rows:
                map_y[i, j] = new_y
            elif new_y < 0:
                # INSTEAD OF RESETTING, CLAMP TO THE TOP EDGE (row 0)
                map_y[i, j] = 0 
            else: # new_y >= rows
                # CLAMP TO THE BOTTOM EDGE
                map_y[i, j] = rows - 1 
            
            # --- Clamping logic for HORIZONTAL ---
            if 0 <= new_x < cols:
                map_x[i, j] = new_x
            elif new_x < 0:
                map_x[i, j] = 0 
            else: 
                map_x[i, j] = cols - 1 
            # --- END OF FIXES ---

    warped_cv_img = cv2.remap(cv_img, map_x, map_y, interpolation=cv2.INTER_NEAREST)
    
    return Image.fromarray(warped_cv_img)
def draw_cluster_noise(image: Image.Image, text_mask: Image.Image):
    """
    Draws the 2x2 block cluster noise onto the main image.
    Uses the text_mask to *avoid* drawing on the text.
    Uses the CLUSTER_TYPES and BLOCK_SIZE settings.
    DENSITY IS NOW RELATIVE TO IMAGE SIZE.
    """
    IMG_WIDTH, IMG_HEIGHT = image.size
    total_cluster_size = BLOCK_SIZE * 2
    # print(f"Drawing clusters based on relative density...")

    # --- 1. Calculate Total Clusters to Draw ---
    total_pixels = IMG_WIDTH * IMG_HEIGHT
    # Calculate how many clusters to draw based on density setting
    total_clusters_to_draw = int((total_pixels / 10000.0) * TOTAL_DENSITY_PER_10K_PIXELS)
    
    # --- 2. Calculate Total Weight ---
    total_weight = sum(cluster_type['density_weight'] for cluster_type in CLUSTER_TYPES)
    
    if total_weight == 0:
        # print("⚠️ WARNING: Total density_weight is 0. No clusters will be drawn.")
        return image

    # print(f"Image size: {IMG_WIDTH}x{IMG_HEIGHT} ({total_pixels}px)")
    # print(f"Total clusters to draw: {total_clusters_to_draw}")

    dot_layer = Image.new('RGBA', (IMG_WIDTH, IMG_HEIGHT), (0, 0, 0, 0))
    dot_drawer = ImageDraw.Draw(dot_layer)
    
    total_clusters_drawn = 0
    for cluster_type in CLUSTER_TYPES:
        
        color1_rgba = ImageColor.getrgb(cluster_type['color1']) + (255,)
        color2_rgba = ImageColor.getrgb(cluster_type['color2']) + (255,)
        
        # --- 3. Calculate count for *this type* ---
        weight_proportion = cluster_type['density_weight'] / total_weight
        density_count = int(weight_proportion * total_clusters_to_draw)
        
        for _ in range(density_count):
            tlx = random.randint(0, IMG_WIDTH - total_cluster_size)
            tly = random.randint(0, IMG_HEIGHT - total_cluster_size)
            
            # Top-Left block (color1)
            dot_drawer.rectangle(
                [tlx, tly, tlx + BLOCK_SIZE - 1, tly + BLOCK_SIZE - 1],
                fill=color1_rgba
            )
            # Top-Right block (color2)
            dot_drawer.rectangle(
                [tlx + BLOCK_SIZE, tly, tlx + total_cluster_size - 1, tly + BLOCK_SIZE - 1],
                fill=color2_rgba
            )
            # Bottom-Left block (color2)
            dot_drawer.rectangle(
                [tlx, tly + BLOCK_SIZE, tlx + BLOCK_SIZE - 1, tly + total_cluster_size - 1],
                fill=color2_rgba
            )
            # Bottom-Right block (color1)
            dot_drawer.rectangle(
                [tlx + BLOCK_SIZE, tly + BLOCK_SIZE, tlx + total_cluster_size - 1, tly + total_cluster_size - 1],
                fill=color1_rgba
            )
        
        total_clusters_drawn += density_count

    # print(f"Total clusters actually drawn: {total_clusters_drawn}")
    del dot_drawer
    
    # --- MODIFICATION: "Cut out" the text shape from the dot layer ---
    # This pastes full transparency (0,0,0,0) *onto* the dot layer,
    # *using* the text mask as the stencil.
    dot_layer.paste((0, 0, 0, 0), (0, 0), text_mask)
    # --- END MODIFICATION ---

    # Paste the transparent dot layer (which now has text-shaped holes)
    image.paste(dot_layer, (0, 0), dot_layer)
    return image


def draw_line_noise(draw: ImageDraw.Draw, IMG_WIDTH: int, IMG_HEIGHT: int):
    """
    Draws the foreground line noise.
    Uses LINE_ settings.
    """
    # print("Drawing line noise...")
    num_lines = random.randint(NUM_LINES_MIN, NUM_LINES_MAX)
    for _ in range(num_lines):
        x1 = random.randint(0, IMG_WIDTH)
        y1 = random.randint(0, IMG_HEIGHT)
        
        length = random.randint(LINE_LENGTH_MIN, LINE_LENGTH_MAX)
        angle = random.uniform(0, 2 * math.pi) # radians

        x2 = int(x1 + length * math.cos(angle))
        y2 = int(y1 + length * math.sin(angle))
        
        line_color_hex = random.choice(LINE_COLORS)
        
        # --- MODIFICATION: Convert hex color to RGBA using new alpha ---
        line_color_rgb = ImageColor.getrgb(line_color_hex)
        line_color_rgba = line_color_rgb + (LINE_NOISE_ALPHA,)
        # --- END MODIFICATION ---

        draw.line([(x1, y1), (x2, y2)], fill=line_color_rgba, width=LINE_WIDTH)


def draw_captcha_text(image_size: tuple, captcha_text: str):
    """
    Draws the distorted text characters onto a NEW transparent layer.
    Returns this layer and the silhouette mask.
    """
    # print(f"Drawing text: {captcha_text}...")
    # --- FIX 1: Use the 'image_size' tuple directly ---
    IMG_WIDTH, IMG_HEIGHT = image_size 
    
    # --- 1. Calculate Positions ---
    font_size_min = int(IMG_HEIGHT * FONT_SIZE_MIN_PERCENT)
    font_size_max = int(IMG_HEIGHT * FONT_SIZE_MAX_PERCENT)
    
    word_base_y = TEXT_START_Y
    left_padding = TEXT_START_X
    jitter_range = int(50 * JITTER_RANGE_PERCENT) 
    current_x_pos = left_padding

    # --- NEW: Create a blank mask AND a blank canvas for the text ---
    # --- FIX 2: Use the 'image_size' tuple directly ---
    text_silhouette_mask = Image.new('L', image_size, 0)
    text_canvas = Image.new('RGBA', image_size, (0, 0, 0, 0)) # <-- Draw on this

    # --- 2. Draw each character ---
    for i, char in enumerate(captcha_text):
        try:
            # 2a. Get random properties
            font_path = random.choice(FONT_FILES)
            font_size = random.randint(font_size_min, font_size_max)
            font = ImageFont.truetype(font_path, size=font_size)
            char_color = random.choice(TEXT_COLORS)
            
            # 2b. Create a transparent layer for the character
            max_distortion = max(CHAR_PIXEL_AMPLITUDE, CHAR_PIXEL_AMPLITUDE_VERTICAL)
            rotation_padding = int(font_size * 0.4)
            padding_buffer_char = int(max_distortion) + rotation_padding
            layer_size = font_size + (padding_buffer_char * 2)
            
            char_img = Image.new('RGBA', (layer_size, layer_size), (0, 0, 0, 0))
            char_draw = ImageDraw.Draw(char_img)
            
            # 1. Create a temporary 1-bit mask to find the character shape
            char_mask = Image.new('L', (layer_size, layer_size), 0)
            mask_draw = ImageDraw.Draw(char_mask)
            mask_draw.text((padding_buffer_char, padding_buffer_char), char, font=font, fill=255) # Draw solid white
            del mask_draw

            # 2. Convert mask to numpy array and find all (y, x) coords of the character
            mask_data = np.array(char_mask)
            y_coords, x_coords = np.where(mask_data > 128) # Find all white pixels
            valid_coords = list(zip(x_coords, y_coords)) # Combine into (x, y) tuples

            if valid_coords:
                # 3. Calculate how many pixels to draw based on density
                num_pixels_to_draw = int(len(valid_coords) * CHAR_PIXEL_DENSITY)
                
                # 4. Randomly sample from the valid coordinates
                pixels_to_draw = random.sample(valid_coords, num_pixels_to_draw)
                
                # 5. Draw *only* the sampled pixels onto the *real* character layer
                for x, y in pixels_to_draw:
                    char_draw.point((x, y), fill=char_color)
            
            del char_draw

            # 2c. Apply Independent Distortions (Shear, Rotate, Wiggle)
            
            if i == 0: # First character
                shear_angle = FIRST_CHAR_SHEAR_ANGLE
            elif i == 1: # Second character
                shear_angle = SECOND_CHAR_SHEAR_ANGLE
            else: # All other characters
                shear_angle = random.uniform(SHEAR_ANGLE_MIN, SHEAR_ANGLE_MAX)

            shear_rad = math.radians(shear_angle)
            shear_matrix = (1, math.tan(shear_rad), 0, 0, 1, 0)
            
            char_img = char_img.transform(
                char_img.size,
                Image.AFFINE,
                shear_matrix,
                resample=Image.NEAREST
            )

            if i == 0: # First character
                angle = FIRST_CHAR_ROTATION_ANGLE
            elif i == 1: # Second character
                angle = SECOND_CHAR_ROTATION_ANGLE
            else: # All other characters
                angle = random.uniform(ROTATION_ANGLE_MIN, ROTATION_ANGLE_MAX)
            
            rotated_img = char_img.rotate(angle, expand=True, resample=Image.NEAREST)
            
            if i == 0: # First character
                current_v_amplitude = FIRST_CHAR_VERTICAL_AMPLITUDE_OVERRIDE
                current_pinch_mode = FIRST_CHAR_VERTICAL_PINCH_MODE
                current_h_amplitude = FIRST_CHAR_HORIZONTAL_AMPLITUDE_OVERRIDE
            elif i == 1: # Second character
                current_v_amplitude = SECOND_CHAR_VERTICAL_AMPLITUDE_OVERRIDE
                current_pinch_mode = "DEFAULT" 
                current_h_amplitude = SECOND_CHAR_HORIZONTAL_AMPLITUDE_OVERRIDE
            else: # All other characters
                current_v_amplitude = CHAR_PIXEL_AMPLITUDE_VERTICAL
                current_pinch_mode = "DEFAULT"
                current_h_amplitude = CHAR_PIXEL_AMPLITUDE
            
            warped_img = apply_wave_warp(
                rotated_img, 
                current_h_amplitude, 
                CHAR_PIXEL_FREQUENCY, 
                current_v_amplitude,
                current_pinch_mode
            )

            # 2d. Apply Global Positioning (Wave + Jitter) and Paste
            base_x = int(current_x_pos)
            
            char_x_relative = base_x - left_padding
            
            if WORD_WAVE_FREQUENCY == 0:
                wave_y_shift = 0
            else:
                wave_value = 1 - math.cos(2 * math.pi * char_x_relative / WORD_WAVE_FREQUENCY)
                wave_y_shift = int((WORD_WAVE_AMPLITUDE / 2.0) * wave_value)
                
            v_jitter = random.randint(-VERTICAL_JITTER_RANGE_PIXELS, VERTICAL_JITTER_RANGE_PIXELS)
            
            # --- THIS IS THE MODIFIED CODE ---
            base_y = word_base_y
            current_y = base_y + wave_y_shift + v_jitter
            
            if i == 0: # i==0 is the *first* character
                current_y += FIRST_CHAR_VERTICAL_OFFSET
            elif i == 1: # i==1 is the *second* character
                current_y += SECOND_CHAR_VERTICAL_OFFSET
            # --- END OF MODIFICATION ---
            
            current_x = base_x + random.randint(-jitter_range, jitter_range)
            
            bbox = warped_img.getbbox()

            if not bbox:
                continue

            paste_x = current_x - bbox[0]
            paste_y = current_y - bbox[1]

            # 5. Paste onto main canvas
            # --- MODIFICATION: Paste onto text_canvas, not the main image ---
            text_canvas.paste(warped_img, (paste_x, paste_y), warped_img)
            # --- END MODIFICATION ---

            # --- Paste onto the silhouette mask (this is unchanged) ---
            try:
                char_silhouette = warped_img.getchannel('A')
                text_silhouette_mask.paste(char_silhouette, (paste_x, paste_y), char_silhouette)
            except Exception as e:
                print(f"    Warning: Could not get alpha mask for char '{char}'. {e}")

            # --- MODIFICATION: Update current_x_pos for the NEXT character ---
            char_width = bbox[2] - bbox[0]
            
            if i == 0: # This was the first character
                current_x_pos = base_x + char_width + CHAR_GAP_1_TO_2_OVERRIDE
            else: # This was character 2, 3, 4...
                current_x_pos = base_x + char_width + CHAR_GAP_DEFAULT
            # --- END MODIFICATION ---

        except Exception as e:
            print(f"Error drawing character '{char}' with font '{font_path}'. Skipping.")
            print(f"Details: {e}")
            
    # --- MODIFICATION: Return the new text_canvas and the mask ---
    return text_canvas, text_silhouette_mask
    # --- END MODIFICATION ---

def generate_captcha(text_to_use=None, output_filename=None, quiet_mode=False):
    """
    Generates a single CAPTCHA image by calling the refactored helper functions.
    
    Args:
        text_to_use (str, optional): The text to render. If None, uses defaults.
        output_filename (str, optional): The path to save the file. 
                                         If None, saves as 'generated_captcha.png' and shows.
        quiet_mode (bool, optional): If True, suppresses print statements.
    """
    
    def log(message):
        if not quiet_mode:
            print(message)

    # --- Step 1: Generate Text & Determine Canvas Size ---
    if text_to_use:
        captcha_text = text_to_use
    elif CAPTCHA_TEXT:
        captcha_text = CAPTCHA_TEXT
    else:
        captcha_text = "".join(random.choice(CHAR_SET) for _ in range(CAPTCHA_LENGTH))

    IMG_HEIGHT = AUTO_SCALE_BASE_HEIGHT
    
    # --- MODIFICATION: New canvas width calculation ---
    if USE_FIXED_CANVAS_WIDTH:
        # Toggle is ON: Use the fixed width
        IMG_WIDTH = FIXED_CANVAS_WIDTH
    else:
        # Toggle is OFF: Use the existing auto-sizing logic
        content_width = 0
        num_chars = len(captcha_text)
        
        if num_chars == 1:
            content_width = CHAR_GAP_DEFAULT 
        elif num_chars > 1:
            # This is a ROUGH approximation for the canvas width.
            # It assumes an "average" character width.
            avg_char_width = 40 # Guess an average width
            
            content_width += avg_char_width # First char
            content_width += CHAR_GAP_1_TO_2_OVERRIDE # First gap
            
            # Add all other chars and gaps
            content_width += (num_chars - 1) * (avg_char_width + CHAR_GAP_DEFAULT)
            
        IMG_WIDTH = content_width + TEXT_START_X + TEXT_START_X
    
    # --- Step 2: Create Canvas & Background Tint ---
    log(f"Creating canvas with size {IMG_WIDTH}x{IMG_HEIGHT}...")
    image = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), 'white')
    
    tint_color = ImageColor.getrgb(BACKGROUND_TINT_COLOR) + (BACKGROUND_TINT_ALPHA,)
    tint_layer = Image.new('RGBA', (IMG_WIDTH, IMG_HEIGHT), tint_color)
    image.paste(tint_layer, (0, 0), tint_layer)
    
    # --- Step 3: Draw Distorted Text (and get the text mask) ---
    text_layer, text_mask = draw_captcha_text((IMG_WIDTH, IMG_HEIGHT), captcha_text)
    if APPLY_CHAR_BLUR:
        log("Applying character-only blur...")
        char_blur_radius = random.uniform(CHAR_BLUR_RADIUS_MIN, CHAR_BLUR_RADIUS_MAX)
        
        # 1. Blur *only* the text layer
        text_layer = text_layer.filter(ImageFilter.GaussianBlur(radius=char_blur_radius))
        
        # 2. CRITICAL: Update the text_mask to match the blurred text
        # This creates the soft edge we need for the noise.
        text_mask = text_layer.getchannel('A')


    # --- NEW STEP 3C: Paste the (possibly blurred) text onto the main image ---
    image.paste(text_layer, (0, 0), text_layer)

    # --- Step 4: Draw Cluster Dot Noise (FOREGROUND) ---
    # Pass the text_mask in to cut out the text shape
    image = draw_cluster_noise(image, text_mask)
    
    # --- Step 4B: (NEW) Draw Noise ON TOP of text ---
    # This draws the *additional* controllable noise (e.g., black/white salt)
    # --- Step 4B: (NEW) Draw CLUSTER Noise ON TOP of text ---
    if APPLY_CHAR_CLUSTER_NOISE:
        image = draw_char_cluster_noise(image, text_mask)


    # --- Step 5: Draw Line Noise (FOREGROUND) ---
    
    # --- MODIFICATION: Draw lines on a separate transparent layer ---
    # 1. Create a new transparent layer
    line_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
    line_drawer = ImageDraw.Draw(line_layer)
    
    # 2. Draw the semi-transparent lines onto it
    draw_line_noise(line_drawer, IMG_WIDTH, IMG_HEIGHT)
    del line_drawer
    
    # 3. Paste the transparent line layer onto the main image
    image.paste(line_layer, (0, 0), line_layer)
    # --- END MODIFICATION ---

    # --- Step 6: Apply Final Effects ---
    if APPLY_GAUSSIAN_BLUR:
        log("Applying Gaussian Blur...")
        blur_radius = random.uniform(GAUSSIAN_BLUR_RADIUS_MIN, GAUSSIAN_BLUR_RADIUS_MAX)
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    if FINAL_ROTATION_ANGLE != 0.0:
        log("Applying final rotation...")
        final_angle = random.uniform(-FINAL_ROTATION_ANGLE, FINAL_ROTATION_ANGLE)
        image = image.rotate(final_angle, resample=Image.NEAREST, expand=True, fillcolor='white')
    
    # --- Step 7: Save and show the final image ---
    if output_filename:
        save_path = output_filename
    else:
        save_path = 'generated_captcha.png'
        
    image.save(save_path)
    
    if not output_filename:
        image.show()
        
    return save_path


# =============================================================================
# 🆕 NEW: DATASET GENERATION FUNCTIONS
# =============================================================================

def setup_dataset_directory(out_root: str, split_names: list, labels_csv_name: str):
    """
    Creates the dataset directory structure and the initial labels CSV file.
    """
    print("=" * 60)
    print("Setting up CAPTCHA Dataset Directory Structure")
    print("=" * 60)
    
    # 1. Create output directories
    try:
        os.makedirs(out_root, exist_ok=True)
        print(f"[INFO] Ensured root directory exists: {out_root}")
        
        for subset in split_names:
            subset_path = os.path.join(out_root, subset)
            os.makedirs(subset_path, exist_ok=True)
            print(f"[INFO] Ensured sub-directory exists: {subset_path}")
            
    except OSError as e:
        print(f"[ERROR] Failed to create directories: {e}")
        return False

    # 2. Create the labels.csv file and write the header
    csv_path = os.path.join(out_root, labels_csv_name)
    
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label"])
        
        print(f"[INFO] Created labels file with header at: {csv_path}")
        
    except IOError as e:
        print(f"[ERROR] Failed to create labels CSV file: {e}")
        return False

    print("=" * 60)
    print("[SUCCESS] Dataset directory structure is ready.")
    print("=" * 60)
    return True

def run_dataset_generation(total_images: int, out_root: str, split_ratio: tuple, labels_list: list):
    """
    Generates a complete dataset with train/val/test splits.
    """
    
    # 1. Setup folders and CSV
    if not setup_dataset_directory(out_root, SPLIT_NAMES, LABELS_CSV_NAME):
        print("[ERROR] Directory setup failed. Aborting generation.")
        return
        
    # 2. Check fonts
    if not check_fonts():
        print("[ERROR] Font check failed. Aborting generation.")
        return
        
    if not labels_list:
        print("[ERROR] `labels_list` is empty. Aborting generation.")
        return
        
    print(f"Generating {total_images} images from {len(labels_list)} unique labels...")
    
    # 3. Calculate split boundaries
    cut_train = int(total_images * split_ratio[0])
    cut_val = cut_train + int(total_images * split_ratio[1])
    
    # 4. Open CSV file to append data
    csv_path = os.path.join(out_root, LABELS_CSV_NAME)
    
    try:
        with open(csv_path, "a", newline="") as f: # 'a' for append
            writer = csv.writer(f)
            pbar = tqdm(total=total_images, unit="img", desc="Generating Dataset")
            
            # 5. Start generation loop
            for counter in range(total_images):
                
                # Get a random label
                label = random.choice(labels_list)
                
                # Determine split
                if counter < cut_train:
                    split = "train"
                elif counter < cut_val:
                    split = "val"
                else:
                    split = "test"
                
                # Define file paths
                filename = f"{counter:06d}_{label}.png"
                full_out_path = os.path.join(out_root, split, filename)
                csv_relative_path = os.path.join(split, filename) # Path to save in CSV
                
                try:
                    # 6. Generate the single captcha image
                    generate_captcha(
                        text_to_use=label, 
                        output_filename=full_out_path, 
                        quiet_mode=True # Suppress logs for speed
                    )
                    
                    # 7. Write to CSV
                    writer.writerow([csv_relative_path, label])
                    
                except Exception as e:
                    print(f"\n[WARNING] Failed to generate image {counter} ({label}): {e}")
                
                pbar.update(1)
            
            pbar.close()
            
    except IOError as e:
        print(f"[ERROR] Could not write to CSV file: {e}")
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}")
    
    print("=" * 60)
    print(f"[SUCCESS] Successfully generated {total_images} CAPTCHA images")
    print(f"[OUTPUT] Output directory: {out_root}")
    print(f"[CSV] Labels CSV: {csv_path}")
    print("=" * 60)


# -----------------------------------------------------------------------------
# 🆕 NEW: Main Control Function (Handles Arguments)
# -----------------------------------------------------------------------------
def main():
    """
    Main function to handle command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate synthetic CAPTCHA images.")
    
    # --- Mode Arguments (choose one) ---
    parser.add_argument("--total", type=int, default=None, 
                       help=f"Generate a full dataset of N images (e.g., 1000).")
    parser.add_argument("--single", type=str, default=None,
                       help="Generate a single test image with specified text (e.g., 'TEST').")

    # --- Dataset-only Arguments ---
    parser.add_argument("--out", type=str, default=DEFAULT_OUT_ROOT,
                       help=f"Output directory for dataset (default: {DEFAULT_OUT_ROOT})")
    parser.add_argument("--split", type=float, nargs=3, default=SPLIT_RATIO,
                       help=f"Train/Val/Test split ratio (default: {' '.join(map(str, SPLIT_RATIO))})")
    
    args = parser.parse_args()
    
    # --- Logic to decide which mode to run ---
    
    if args.total and args.total > 0:
        # --- DATASET GENERATION MODE ---
        print("[MODE] Running Dataset Generation...")
        # Validate split ratio
        if abs(sum(args.split) - 1.0) > 0.001:
            print(f"[ERROR] Split ratios must sum to 1.0. Got: {sum(args.split)}")
            return
            
        run_dataset_generation(
            total_images=args.total,
            out_root=args.out,
            split_ratio=tuple(args.split),
            labels_list=DEFAULT_LABELS_LIST
        )
        
    elif args.single:
        # --- SINGLE IMAGE TEST MODE ---
        print(f"[MODE] Running Single Image Test for: '{args.single}'")
        if check_fonts():
            output_file = f"test_{args.single}.png"
            generate_captcha(text_to_use=args.single, output_filename=output_file)
            print(f"\n🎉 Generated '{output_file}' successfully!")
            # Open the image
            try:
                Image.open(output_file).show()
            except Exception:
                pass # Ignore if 'show' fails
        
    else:
        # --- DEFAULT (ORIGINAL) TEST MODE ---
        print("[MODE] Running in default standalone test mode...")
        if check_fonts():
            print("Generating test 'BTC.png'...")
            generate_captcha(text_to_use="BTC", output_filename="BTC_generated.png")
            print("\n🎉 Generated 'BTC_generated.png' successfully!")
            # Open the image
            try:
                Image.open("BTC_generated.png").show()
            except Exception:
                pass # Ignore if 'show' fails


if __name__ == "__main__":
    main()