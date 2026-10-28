"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    result = np.zeros((matrices[0].shape[0], 1), dtype=float)
    for A, x in zip(matrices, vectors):
        result += A @ x
    return result


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (matrix > threshold).astype(float)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [np.unique(col).tolist() for col in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))
    row_means = np.mean(matrix, axis=1)
    column_means = np.mean(matrix, axis=0)
    row_variances = np.var(matrix, axis=1)
    column_variances = np.var(matrix, axis=0)
    return MatrixStatistics(
        matrix=matrix,
        row_means=row_means,
        column_means=column_means,
        row_variances=row_variances,
        column_variances=column_variances,
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    matrix = np.empty((rows, columns), dtype=float)
    for i in range(rows):
        for j in range(columns):
            if (i + j) % 2 == 0:
                matrix[i, j] = first
            else:
                matrix[i, j] = second
    return matrix


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.full((image_height, image_width, 3), background_color, dtype=int)
    # Center the rectangle
    start_y = (image_height - height) // 2
    start_x = (image_width - width) // 2
    end_y = start_y + height
    end_x = start_x + width
    image[start_y:end_y, start_x:end_x] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    image = np.full((image_height, image_width, 3), background_color, dtype=int)
    y0 = image_height / 2.0
    x0 = image_width / 2.0
    y, x = np.ogrid[:image_height, :image_width]
    # (x - x0)^2 / a^2 + (y - y0)^2 / b^2 <= 1
    mask = ((x - x0) ** 2) / (semi_axis_x ** 2) + ((y - y0) ** 2) / (semi_axis_y ** 2) <= 1
    image[mask] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    variance = float(np.var(values))
    std = float(np.std(values))
    # Strict local maxima and minima
    local_maxima = []
    local_minima = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            local_maxima.append(i)
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            local_minima.append(i)
    # Moving average
    if window > len(values) or window < 1:
        moving_average = np.array([])
    else:
        moving_average = np.convolve(values, np.ones(window) / window, mode="valid")
    return TimeSeriesStatistics(
        mean=mean,
        variance=variance,
        std=std,
        local_maxima_indices=np.array(local_maxima),
        local_minima_indices=np.array(local_minima),
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    labels = np.asarray(labels)
    if class_count is None:
        class_count = int(np.max(labels)) + 1 if len(labels) > 0 else 0
    result = np.zeros((len(labels), class_count), dtype=int)
    if len(labels) > 0:
        result[np.arange(len(labels)), labels] = 1
    return result