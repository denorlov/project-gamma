import cv2
import numpy as np


def load_and_process_image(img_path):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray_img


def normalize_matrix(matrix):
    max_val = np.max(matrix)
    normalized = matrix - (max_val / 2)
    return normalized


def compute_prefix_sum(matrix):
    prefix_sum = np.zeros(matrix.shape)
    prefix_sum[0, 0] = matrix[0, 0]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            prefix_sum[i, j] = matrix[i, j]
            if i > 0:
                prefix_sum[i, j] += prefix_sum[i - 1, j]
            if j > 0:
                prefix_sum[i, j] += prefix_sum[i, j - 1]
            if i > 0 and j > 0:
                prefix_sum[i, j] -= prefix_sum[i - 1, j - 1]
    return prefix_sum


def find_top_n_max_sum_submatrices(normalized_matrix, n=5):
    nrows, ncols = normalized_matrix.shape
    prefix_sum = compute_prefix_sum(normalized_matrix)
    results = []

    for r1 in range(nrows):
        for r2 in range(r1, nrows):
            current_sum = 0
            start_col = 0
            for c in range(ncols):
                submatrix_sum = prefix_sum[r2, c]
                if r1 > 0:
                    submatrix_sum -= prefix_sum[r1 - 1, c]
                if start_col > 0:
                    submatrix_sum -= prefix_sum[r2, start_col - 1]
                if r1 > 0 and start_col > 0:
                    submatrix_sum += prefix_sum[r1 - 1, start_col - 1]

                current_sum += submatrix_sum
                if current_sum > 0:
                    results.append((current_sum, (r1, start_col), (r2, c)))
                if current_sum < 0:
                    current_sum = 0
                    start_col = c + 1

    # Сортируем результаты по убыванию суммы и берем n наилучших
    results.sort(reverse=True, key=lambda x: x[0])

    # Фильтруем результаты по расстоянию и пересечению
    filtered_results = []
    for result in results:
        _, top_left, bottom_right = result
        width = bottom_right[1] - top_left[1] + 1
        height = bottom_right[0] - top_left[0] + 1

        # Проверка на пересечение с уже добавленными прямоугольниками
        intersection_found = False
        for other in filtered_results:
            _, other_top_left, other_bottom_right = other
            other_width = other_bottom_right[1] - other_top_left[1] + 1
            other_height = other_bottom_right[0] - other_top_left[0] + 1

            # Проверка на пересечение
            if not (top_left[0] > other_bottom_right[0] or bottom_right[0] < other_top_left[0] or
                    top_left[1] > other_bottom_right[1] or bottom_right[1] < other_top_left[1]):
                intersection_found = True
                break

            # Проверка на расстояние
            if np.linalg.norm(np.array(top_left) - np.array(other[1])) <= max(width, height, other_width, other_height):
                intersection_found = True
                break

        if not intersection_found:
            filtered_results.append(result)
        if len(filtered_results) >= n:
            break

    return filtered_results[:n]


def draw_rectangles_on_image(original_image, rectangles):
    for _, top_left, bottom_right in rectangles:
        cv2.rectangle(original_image,
                      (top_left[1], top_left[0]),
                      (bottom_right[1], bottom_right[0]),
                      (0, 255, 0), 1)
    return original_image


def main(img_path):
    original_image, gray_image = load_and_process_image(img_path)
    normalized_matrix = normalize_matrix(gray_image)
    top_n_rectangles = find_top_n_max_sum_submatrices(normalized_matrix)
    result_image = draw_rectangles_on_image(original_image.copy(), top_n_rectangles)

    # Изменение размера изображения
    # if width and height:
    #     result_image = cv2.resize(result_image, (width, height))

    cv2.imshow("result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = 'res/test-big-image.png'
main(image_path)