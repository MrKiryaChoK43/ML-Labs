"""тесты для numpy_tasks.py"""

import numpy as np
from numpy_tasks import (
    sum_prod, binarize, unique_rows, unique_columns,
    matrix_statistics, chess, draw_rectangle, draw_ellipse,
    analyze_time_series, one_hot,
)
from grader_contracts.numpy_tasks import (
    MatrixVectorBatchInput, BinarizeInput, MatrixInput,
    RandomMatrixInput, ChessInput, RectangleInput, EllipseInput,
    TimeSeriesInput, OneHotInput,
)


def test_sum_prod():
    A1 = np.array([[1.0, 0], [0, 1]])
    x1 = np.array([[1.0], [2]])
    A2 = np.array([[2.0, 0], [0, 2]])
    x2 = np.array([[3.0], [4]])
    result = sum_prod(MatrixVectorBatchInput([A1, A2], [x1, x2]))
    np.testing.assert_allclose(result, [[7.0], [10.0]])

    # Одна матрица
    result = sum_prod(MatrixVectorBatchInput([A1], [x1]))
    np.testing.assert_allclose(result, [[1.0], [2.0]])

    # Нулевые векторы
    z = np.zeros((2, 1))
    result = sum_prod(MatrixVectorBatchInput([A1, A2], [z, z]))
    np.testing.assert_allclose(result, [[0.0], [0.0]])
    print("sum_prod: OK")


def test_binarize():
    M = np.array([[0.1, 0.6], [0.7, 0.4]])
    result = binarize(BinarizeInput(M, 0.5))
    np.testing.assert_array_equal(result, [[0.0, 1.0], [1.0, 0.0]])

    # Всё ниже / выше порога
    result = binarize(BinarizeInput(M, 1.0))
    np.testing.assert_array_equal(result, [[0.0, 0.0], [0.0, 0.0]])
    result = binarize(BinarizeInput(M, -1.0))
    np.testing.assert_array_equal(result, [[1.0, 1.0], [1.0, 1.0]])

    # Равно порогу → 0 (строго больше)
    M2 = np.array([[0.5, 0.5], [0.5, 0.6]])
    result = binarize(BinarizeInput(M2, 0.5))
    np.testing.assert_array_equal(result, [[0.0, 0.0], [0.0, 1.0]])

    # Исходная матрица не изменилась
    assert M[0, 0] == 0.1
    print("binarize: OK")


def test_unique_rows_columns():
    M = np.array([[1.0, 2, 1], [3, 3, 4], [1, 2, 1]])
    rows = unique_rows(MatrixInput(M))
    assert rows[0] == [1.0, 2.0]
    assert rows[1] == [3.0, 4.0]
    assert rows[2] == [1.0, 2.0]

    cols = unique_columns(MatrixInput(M))
    assert cols[0] == [1.0, 3.0]
    assert cols[1] == [2.0, 3.0]
    assert cols[2] == [1.0, 4.0]

    # Все одинаковые
    M2 = np.ones((2, 3))
    assert unique_rows(MatrixInput(M2))[0] == [1.0]

    # Одна строка
    M3 = np.array([[5.0, 5, 7]])
    assert unique_rows(MatrixInput(M3)) == [[5.0, 7.0]]
    print("unique_rows / unique_columns: OK")


def test_matrix_statistics():
    stats = matrix_statistics(RandomMatrixInput(4, 5, 0.0, 1.0, 42))
    assert stats.matrix.shape == (4, 5)
    assert stats.row_means.shape == (4,)
    assert stats.column_means.shape == (5,)

    # Воспроизводимость по seed
    stats2 = matrix_statistics(RandomMatrixInput(4, 5, 0.0, 1.0, 42))
    np.testing.assert_array_equal(stats.matrix, stats2.matrix)

    # 1x1
    stats4 = matrix_statistics(RandomMatrixInput(1, 1, 5.0, 0.0, 1))
    assert abs(stats4.row_means[0] - 5.0) < 1e-9
    print("matrix_statistics: OK")


def test_chess():
    result = chess(ChessInput(3, 4, 0, 1))
    expected = np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1]], dtype=float)
    np.testing.assert_array_equal(result, expected)

    # 1x1
    result = chess(ChessInput(1, 1, 5, 9))
    np.testing.assert_array_equal(result, [[5.0]])

    # 2x2
    result = chess(ChessInput(2, 2, 1, 0))
    np.testing.assert_array_equal(result, [[1, 0], [0, 1]])
    print("chess: OK")


def test_draw_rectangle():
    img = draw_rectangle(RectangleInput(2, 2, 4, 4, (255,0,0), (0,0,0)))
    assert img.shape == (4, 4, 3)
    assert np.array_equal(img[1, 1], [255, 0, 0])
    assert np.array_equal(img[0, 0], [0, 0, 0])

    # На весь размер
    img2 = draw_rectangle(RectangleInput(3, 3, 3, 3, (1,2,3), (0,0,0)))
    assert np.all(img2 == [1, 2, 3])
    print("draw_rectangle: OK")


def test_draw_ellipse():
    img = draw_ellipse(EllipseInput(1, 1, 5, 5, (255,0,0), (0,0,0)))
    assert img.shape == (5, 5, 3)
    assert np.array_equal(img[2, 2], [255, 0, 0])
    assert np.array_equal(img[0, 0], [0, 0, 0])

    # Большой эллипс
    img3 = draw_ellipse(EllipseInput(10, 10, 5, 5, (7,8,9), (0,0,0)))
    assert np.all(img3 == [7, 8, 9])
    print("draw_ellipse: OK")


def test_analyze_time_series():
    values = np.array([1.0, 3, 2, 5, 4, 6])
    stats = analyze_time_series(TimeSeriesInput(values, 3))
    assert abs(stats.mean - 3.5) < 1e-9
    np.testing.assert_array_equal(stats.local_maxima_indices, [1, 3])
    np.testing.assert_array_equal(stats.local_minima_indices, [2, 4])

    # Строго возрастающий — нет экстремумов
    values2 = np.array([1.0, 2, 3, 4, 5])
    stats2 = analyze_time_series(TimeSeriesInput(values2, 2))
    assert len(stats2.local_maxima_indices) == 0
    assert len(stats2.local_minima_indices) == 0

    # Короткий ряд
    stats3 = analyze_time_series(TimeSeriesInput(np.array([10.0]), 1))
    assert stats3.mean == 10.0
    assert len(stats3.local_maxima_indices) == 0

    # Окно больше длины
    stats4 = analyze_time_series(TimeSeriesInput(np.array([10.0]), 5))
    assert len(stats4.moving_average) == 0
    print("analyze_time_series: OK")


def test_one_hot():
    result = one_hot(OneHotInput([0, 2, 3, 0]))
    expected = np.array([[1,0,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]])
    np.testing.assert_array_equal(result, expected)

    # С class_count
    result = one_hot(OneHotInput([0, 2, 3, 0], class_count=5))
    expected = np.array([[1,0,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[1,0,0,0,0]])
    np.testing.assert_array_equal(result, expected)

    # Один класс
    result = one_hot(OneHotInput([0, 0, 0]))
    np.testing.assert_array_equal(result, [[1],[1],[1]])

    # Пустой
    result = one_hot(OneHotInput([]))
    assert result.shape[0] == 0
    print("one_hot: OK")


if __name__ == "__main__":
    test_sum_prod()
    test_binarize()
    test_unique_rows_columns()
    test_matrix_statistics()
    test_chess()
    test_draw_rectangle()
    test_draw_ellipse()
    test_analyze_time_series()
    test_one_hot()
    print("\nВсе тесты пройдены успешно")