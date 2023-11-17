from PIL import Image, ImageDraw, ImageFont
import os

dir_path = r"C:\Users\pc\Documents\__protein design\SURPASS\Rouse Model\mc002\log-log_wrap_moving_average~2"
image_files = [img_file for img_file in os.listdir(dir_path) if img_file.endswith('.png')]
images = [Image.open(os.path.join(dir_path, img_file)) for img_file in image_files]

image_width, image_height = images[0].size
collage_width = 4 * image_width
collage_height = ((len(images) - 1) // 4 + 1) * image_height

collage = Image.new('RGB', (collage_width, collage_height))
draw = ImageDraw.Draw(collage)

# you may need to download or specify the full path to the font file
font = ImageFont.truetype("arial.ttf", 15)

for i, image in enumerate(images):
    x = i % 4 * image_width
    y = i // 4 * image_height
    collage.paste(image, (x, y))
    draw.text((x, y), image_files[i], fill="white", font=font)

collage.save(os.path.join(dir_path, 'collage.png'))