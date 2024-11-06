import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from PIL.Image import Resampling


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
    baseline = int(max_value * 0.75)
    N = M - baseline
    return N, max_value


def maximumSumRectangle(matrix):
    # print("maximumSumRectangle matrix:")
    # for row in matrix:
    #     print(row)

    max_sum = float('-inf')
    final_left = final_right = final_top = final_bottom = 0

    R = len(matrix)
    C = len(matrix[0])
    print(f"maximumSumRectangle RxC: {R}*{C}")

    for left in range(C):
        print(f"left: {left}")
        left_to_right_sum_in_rows = [0] * R
        for right in range(left, C):
            print(f"left: {left}, right:{right}, RxC={R}x{C}")
            for i in range(R):
                left_to_right_sum_in_rows[i] += matrix[i][right]

            #print(f"left={left}, right={right}, left_to_right_sum_in_rows={left_to_right_sum_in_rows}")

            current_sum = 0
            temp_max = float('-inf')
            temp_top = 0
            temp_bottom = 0

            start_row = 0
            for end_row in range(R):
                current_sum = max(left_to_right_sum_in_rows[end_row], current_sum + left_to_right_sum_in_rows[end_row])
                #print(f"left,right={left},{right}, current_sum={current_sum}")

                if current_sum > temp_max:
                    temp_max = current_sum
                    temp_top = start_row
                    temp_bottom = end_row
                    #print(f"new local max, top,bottom={final_top},{final_bottom}, max: {temp_max}")

                if current_sum < 0:
                    current_sum = 0
                    start_row = end_row + 1
                    #print(f"reset counting, current_sum={current_sum}, new start_row={start_row}")


            if temp_max > max_sum:
                max_sum = temp_max
                final_left = left
                final_right = right
                final_top = temp_top
                final_bottom = temp_bottom
                #print(f"new absolute max, left,right={left},{right}; top,bottom={final_top},{final_bottom}, max={temp_max}")

    return max_sum, (final_top, final_left), (final_bottom, final_right)


def draw_circle_on_image(img, center, radius):
    # Рисуем окружность на изображении
    draw = ImageDraw.Draw(img)
    x, y = center
    bbox = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bbox, outline="red", width=5)

def draw_rect_on_image(img, top_left, bottom_right, scale_factor):
    draw = ImageDraw.Draw(img)
    draw.rectangle(((top_left[1]*scale_factor,top_left[0]*scale_factor), (bottom_right[1]*scale_factor, bottom_right[0]*scale_factor)), outline="red", width=5)



def main(image_path):
    img = load_image(image_path)

    fig = plt.figure()

    gray_img = convert_to_grayscale(img)

    a = fig.add_subplot(2, 1, 1)
    plt.imshow(gray_img)
    a.set_title("grey scale")

    width, height = gray_img.size
    scale_factor = 0.25
    scaled_img = gray_img.resize(size=(int(width * scale_factor), int(height * scale_factor)))

    M = np.array(scaled_img, dtype="int32")
    #M = np.array(gray_img)
    print("gray scale:")
    print(M.dtype)

    # for row in M:
    #     print(row)

    N, max_value = normalize_matrix(M)
    print(f"max_value: {max_value}")

    print("normalised grey scale:")
    print(N.dtype)

    for row in N:
        print(row)

    max_sum, top_left, bottom_right = maximumSumRectangle(N)

    print(max_sum, top_left, bottom_right)
    draw_rect_on_image(img, top_left, bottom_right, 1 / scale_factor)
    # draw_rect_on_image(img, top_left, bottom_right, 1)

    a = fig.add_subplot(2, 1, 2)
    plt.imshow(img)
    a.set_title("brightest area")

    plt.show()

if __name__ == '__main__':
    main('./res/test-small-image.png')
    #main('./res/test-mid-image.png')
    #main('./res/test-big-image.png')
    #main('./res/image3.tif')
    #main('./res/image2.tif')
    #main('./res/image3.tif')
