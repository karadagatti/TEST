# ocr_utils.py
import pytesseract
#from pdf2image import convert_from_path
from PIL import Image, ImageOps, ImageFilter
import cv2
import numpy as np
import os

def correct_rotation(pil_image):
    image = np.array(pil_image)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return Image.fromarray(rotated)

def ocr_image(image_path):
    img = Image.open(image_path)
    img = correct_rotation(img)
    img = img.convert('L')
    img = ImageOps.invert(img)
    img = img.filter(ImageFilter.SHARPEN)
    return pytesseract.image_to_string(img)

def extract_text_from_file(file_path):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
    img = Image.open(file)
    text = pytesseract.image_to_string(img)
        pages = convert_from_path(file_path, dpi=300)
        for page in pages:
            page = correct_rotation(page)
            page = page.convert('L')
            page = ImageOps.invert(page)
            page = page.filter(ImageFilter.SHARPEN)
            text += pytesseract.image_to_string(page) + "\n"
        return text
    else:
        return ocr_image(file_path)
