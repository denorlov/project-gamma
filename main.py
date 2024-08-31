import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def load_image(image_path):
    # Загружаем изображение
    img = Image.open(image_path)
    return img


def convert_to_grayscale(img):
    # Преобразуем изображение в градации серого
    return img.convert('L')


def normalize_matrix(M):
    # Нормализуем матрицу M
    max_value = np.max(M)
    N = M - max_value
    return N, max_value


def maximumSumRectangle(R, C, M):
    max_sum = float('-inf')
    final_left = final_right = final_top = final_bottom = 0

    for left in range(C):
        temp = [0] * R
        for right in range(left, C):
            for i in range(R):
                temp[i] += M[i][right]

            current_sum = 0
            max_temp = float('-inf')
            start_row = 0

            for end_row in range(R):
                current_sum = max(temp[end_row], current_sum + temp[end_row])
                if current_sum > max_temp:
                    max_temp = current_sum
                    final_top = start_row
                    final_bottom = end_row

                if current_sum < 0:
                    current_sum = 0
                    start_row = end_row + 1

            if max_temp > max_sum:
                max_sum = max_temp
                final_left = left
                final_right = right

    return max_sum, (final_top, final_left), (final_bottom, final_right)


def draw_circle_on_image(img, center, radius):
    # Рисуем окружность на изображении
    draw = ImageDraw.Draw(img)
    x, y = center
    bbox = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bbox, outline="red", width=1)

def draw_rect_on_image(img, top_left, bottom_right):
    # Рисуем окружность на изображении
    draw = ImageDraw.Draw(img)
    draw.rectangle((top_left, bottom_right), outline="red", width=1)



def main(image_path):
    img = load_image(image_path)
    gray_img = convert_to_grayscale(img)

    M = np.array(gray_img)
    print("gray scale:")
    print(M.dtype)
    for row in M:
        print(row)

    N, max_value = normalize_matrix(M)
    print(f"max_value: {max_value}")

    print("normalised gray scale:")
    print(N.dtype)
    for row in N:
        print(row)

    max_sum, top_left, bottom_right = maximumSumRectangle(*N.shape, N)

    print(max_sum, top_left, bottom_right)
    draw_rect_on_image(img, top_left, bottom_right)

    plt.imshow(img)
    plt.axis('off')
    plt.show()


#main('./res/image.jpg')
main('./res/test-small-image.png')