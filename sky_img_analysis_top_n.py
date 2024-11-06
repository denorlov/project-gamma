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
    baseline = int(max_value * 0.85)
    N = M - baseline
    return N, max_value


class Rect:
    def __init__(self, left, right, top, bottom):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def __repr__(self):
        return f"[left_top={self.left},{self.top}; right_bottom={self.right}, {self.bottom}]"

    def intersect(self, other):
        return (self.right >= other.left and self.bottom <= other.top) or\
            (other.right >= self.left and other.bottom <= self.top) or\
            (self.right > other.left and self.top > other.bottom) or\
            (other.right > self.left and other.top > self.bottom)

    def max_side_length(self):
        return max(
            abs(self.top - self.bottom),
            abs(self.left - self.right),
        )

    def center(self):
        return np.array((int((self.right - self.left) / 2), int((self.top - self.bottom) / 2)))


class Submatrix:
    def __init__(self, bounds: Rect, total_sum):
        self.bounds = bounds
        self.total_sum = total_sum

    def __init__(self, left, right, top, bottom, total_sum):
        self.bounds = Rect(left, right, top, bottom)
        self.total_sum = total_sum



    def __repr__(self):
        return f"[bounds={self.bounds}, sum={self.total_sum}]"

    def top_left(self):
        return (self.bounds.top, self.bounds.left)

    def center(self):
        return self.bounds.center()


def maximumSumRectangle(matrix):
    # print("maximumSumRectangle matrix:")
    # for row in matrix:
    #     print(row)

    results:Submatrix = []
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

                # if current_sum > temp_max:
                #     temp_max = current_sum
                #     temp_top = start_row
                #     temp_bottom = end_row
                #     #print(f"new local max, top,bottom={final_top},{final_bottom}, max: {temp_max}")

                results.append(Submatrix(left, right, start_row, end_row, current_sum))
                # had_intersections = False
                # for r1 in results:
                #     if r.bounds.intersect(r1.bounds):
                #         had_intersections = True
                #         if r.total_sum > r1.total_sum:
                #             results.remove(r1)
                #             results.append(r)
                # if not had_intersections:
                #     results.append(r)

                if current_sum < 0:
                    current_sum = 0
                    start_row = end_row + 1
                    #print(f"reset counting, current_sum={current_sum}, new start_row={start_row}")

            # r = Submatrix(left, right, temp_top, temp_bottom, temp_max)
            # had_intersections = False
            # for r1 in results:
            #     if r.bounds.intersect(r1.bounds):
            #         had_intersections = True
            #         if r.total_sum > r1.total_sum:
            #             results.remove(r1)
            #             results.append(r)
            # if not had_intersections:
            #     results.append(r)

            # if temp_max > max_sum:
            #     max_sum = temp_max
            #     final_left = left
            #     final_right = right
            #     final_top = temp_top
            #     final_bottom = temp_bottom
            #     print(f"new absolute max, left,right={left},{right}; top,bottom={final_top},{final_bottom}, max={temp_max}")

    # Сортируем результаты по убыванию суммы и берем n наилучших
    results.sort(key=lambda x: x.total_sum)

    # Фильтруем результаты по расстоянию
    filtered_results = []
    for submatrix in results:
        close_to_filtered = False
        for other_submatrix in filtered_results:
            max_side_length = max(
                submatrix.bounds.max_side_length(),
                other_submatrix.bounds.max_side_length()
            )
            if np.linalg.norm(submatrix.center() - other_submatrix.center()) <= max_side_length:
                close_to_filtered = True
        if not close_to_filtered:
            filtered_results.append(submatrix)
        if len(filtered_results) >= 5:
            break

    return results

def draw_circle_on_image(img, center, radius):
    # Рисуем окружность на изображении
    draw = ImageDraw.Draw(img)
    x, y = center
    bbox = [x - radius, y - radius, x + radius, y + radius]
    draw.ellipse(bbox, outline="red", width=5)


def draw_rect_on_image(img, rect:Rect, scale_factor):
    draw = ImageDraw.Draw(img)
    draw.rectangle(((rect.left * scale_factor, rect.top * scale_factor),
                    (rect.right * scale_factor, rect.bottom * scale_factor)), outline="red", width=5)


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

    submatrixes = maximumSumRectangle(N)
    for smatrix in submatrixes:
        draw_rect_on_image(img, smatrix.bounds, 1 / scale_factor)

    a = fig.add_subplot(2, 1, 2)
    plt.imshow(img)
    a.set_title("brightest area")

    plt.show()


if __name__ == '__main__':
    #main('./res/test-small-image.png')
    #main('./res/test-mid-image.png')
    #main('./res/test-big-image.png')
    main('./res/image1.tif')
    #main('./res/image2.tif')
    #main('./res/image3.tif')
