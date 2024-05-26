import cv2
import numpy as np

def load_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

def preprocess_image(gray_image):
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def find_cards(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cards = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # 사각형 모양만 선택
            cards.append(approx)
    return cards

def crop_and_save_cards(image, cards, output_prefix='card'):
    for i, card in enumerate(cards):
        x, y, w, h = cv2.boundingRect(card)
        cropped_card = image[y:y+h, x:x+w]
        cv2.imwrite(f"{output_prefix}_{i}.png", cropped_card)

def main(image_path):
    image, gray_image = load_image(image_path)
    edges = preprocess_image(gray_image)
    cards = find_cards(edges)
    crop_and_save_cards(image, cards)

# 사용 예제
main('/workspace/Sutda_test/20240522_143207.jpg')
