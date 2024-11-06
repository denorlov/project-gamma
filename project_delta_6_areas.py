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

    # Фильтруем результаты по расстоянию
    filtered_results = []
    for result in results:
        rect_sum, top_left, bottom_right = result
        close_to_filtered = False
        for other in filtered_results:
            other_rect_sum, other_top_left, other_bottom_right = result
            max_side_length = max(
                abs(other_top_left[0] - other_bottom_right[0]),
                abs(other_top_left[1] - other_bottom_right[1]),
                abs(top_left[0] - bottom_right[0]),
                abs(top_left[1] - bottom_right[1]),
            )
            if np.linalg.norm(np.array(top_left) - np.array(other[1])) <= max_side_length:
                close_to_filtered = True
        if not close_to_filtered:
            filtered_results.append(result)
        if len(filtered_results) >= n:
            break

    return filtered_results[:n]


def draw_rectangles_on_image(original_image, rectangles):
    for i, r in enumerate(rectangles):
        _, top_left, bottom_right = r
        cv2.rectangle(original_image,
                      (top_left[1], top_left[0]),
                      (bottom_right[1], bottom_right[0]),
                      (0, 0, 128 + i * 20), 1)
    return original_image


def main(img_path, width=False, height=False):
    original_image, gray_image = load_and_process_image(img_path)
    normalized_matrix = normalize_matrix(gray_image)
    top_n_rectangles = find_top_n_max_sum_submatrices(normalized_matrix)
    result_image = draw_rectangles_on_image(original_image.copy(), top_n_rectangles)

    # Изменение размера изображения
    if width and height:
        result_image = cv2.resize(result_image, (width, height))

    cv2.imshow("result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = 'res/test-big-image.png'
main(image_path)
#main(image_path, 173 * 5, 44 * 5)
