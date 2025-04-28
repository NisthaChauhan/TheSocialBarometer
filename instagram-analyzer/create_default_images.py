from PIL import Image, ImageDraw
import os

def create_default_image(size, color, text, output_path):
    """Create a default image with specified size, color and text"""
    img = Image.new('RGB', size, color)
    draw = ImageDraw.Draw(img)
    
    # Add text
    text_width = draw.textlength(text)
    text_height = 20
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    draw.text(position, text, fill='white')
    
    # Save image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)

# Create default images
static_dir = os.path.join('Nistha', 'Insta', 'WORKING', 'static')

create_default_image((400, 400), (200, 200, 200), 'Default Post', os.path.join(static_dir, 'default-post.png'))
create_default_image((150, 150), (150, 150, 150), 'Default Profile', os.path.join(static_dir, 'default-profile.png')) 