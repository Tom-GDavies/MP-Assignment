#####################################################
# Python program to generate synthetic data to train CNN
#####################################################

from PIL import Image, ImageDraw, ImageFont
import os
import shutil
import numpy as np
import cv2
import random

output_dir = "synthetic_digits"

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

fonts = ["Baloo2-Regular.ttf"]
digits = "0123456789"

for digit in digits:
    for i in range(100):
        img = Image.new("RGB", (200,200), color=(255,255,255))
        draw = ImageDraw.Draw(img)

        font_path = random.choice(fonts)
        font_size = random.randint(160,200)
        font = ImageFont.truetype(font_path,font_size)

        bbox = draw.textbbox((0, 0), digit, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        x = (200 - w) // 2 - bbox[0] + random.randint(-10, 10)
        y = (200 - h) // 2 - bbox[1] + random.randint(-10, 10)

        draw.text((x, y), digit, font=font, fill=(0,0,0))



        img_np = np.array(img)
        angle = random.uniform(-15,15)
        M = cv2.getRotationMatrix2D((100,100), angle, 1)
        img_np = cv2.warpAffine(img_np, M, (200,200), borderValue=(255,255,255))

        img_final = Image.fromarray(img_np)
        img_final.save(os.path.join(output_dir, f"{digit}_{i}.png"))