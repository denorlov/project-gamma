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


def find_max_sum_submatrix(normalized_matrix):
    nrows, ncols = normalized_matrix.shape
    prefix_sum = compute_prefix_sum(normalized_matrix)

    max_sum = float('-inf')
    top_left = (0, 0)
    bottom_right = (0, 0)

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

                if current_sum > max_sum:
                    max_sum = current_sum
                    top_left = (r1, start_col)
                    bottom_right = (r2, c)

                if current_sum < 0:
                    current_sum = 0
                    start_col = c + 1

    return top_left, bottom_right


def draw_rectangle_on_image(original_image, top_left, bottom_right):
    cv2.rectangle(original_image,
                  (top_left[1], top_left[0]),
                  (bottom_right[1], bottom_right[0]),
                  (0, 255, 0), 2)
    return original_image


def main(img_path):
    original_image, gray_image = load_and_process_image(img_path)
    normalized_matrix = normalize_matrix(gray_image)

    top_left, bottom_right = find_max_sum_submatrix(normalized_matrix)
    result_image = draw_rectangle_on_image(original_image.copy(), top_left, bottom_right)

    cv2.imshow("result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main('./res/test-big-image.png')
    #main('./res/test-mid-image.png')
    #main('./res/test-big-image.png')
    #main('./res/image3.tif')
    #main('./res/image2.tif')
    #main('./res/image3.tif')
