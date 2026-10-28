"""Собственные тесты для python_basics.py (включая граничные случаи)."""

from python_basics import (
    count_vowels,
    has_unique_characters,
    count_one_bits,
    multiplicative_persistence,
    mse,
    prime_factorization,
    pyramid,
    is_balanced_number,
)
from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def test_count_vowels():
    # Примеры / обычные случаи
    assert count_vowels(TextInput("Hello")) == 2
    assert count_vowels(TextInput("AEIOUaeiou")) == 10
    assert count_vowels(TextInput("Python")) == 1
    # Граничные
    assert count_vowels(TextInput("")) == 0                    # пустая строка
    assert count_vowels(TextInput("bcdfg")) == 0               # нет гласных
    assert count_vowels(TextInput("a")) == 1                   # одна гласная
    assert count_vowels(TextInput("A")) == 1                   # верхний регистр
    assert count_vowels(TextInput("xyzXYZ")) == 0
    assert count_vowels(TextInput("rhythm")) == 0              # нет гласных (y не считается)
    print("count_vowels: OK")


def test_has_unique_characters():
    # Обычные
    assert has_unique_characters(TextInput("abc")) is True
    assert has_unique_characters(TextInput("hello")) is False
    assert has_unique_characters(TextInput("Python")) is True
    # Граничные
    assert has_unique_characters(TextInput("")) is True        # пустая строка
    assert has_unique_characters(TextInput("a")) is True       # один символ
    assert has_unique_characters(TextInput("aa")) is False     # повтор
    assert has_unique_characters(TextInput("aA")) is True      # разный регистр — разные символы
    assert has_unique_characters(TextInput("12321")) is False
    assert has_unique_characters(TextInput("!@#$")) is True
    print("has_unique_characters: OK")


def test_count_one_bits():
    # Обычные
    assert count_one_bits(PositiveIntegerInput(13)) == 3      # 1101
    assert count_one_bits(PositiveIntegerInput(1)) == 1       # 1
    assert count_one_bits(PositiveIntegerInput(7)) == 3       # 111
    assert count_one_bits(PositiveIntegerInput(8)) == 1       # 1000
    # Граничные
    assert count_one_bits(PositiveIntegerInput(0)) == 0
    assert count_one_bits(PositiveIntegerInput(2**10)) == 1   # степень двойки
    assert count_one_bits(PositiveIntegerInput(2**10 - 1)) == 10  # все единицы
    assert count_one_bits(PositiveIntegerInput(255)) == 8
    print("count_one_bits: OK")


def test_multiplicative_persistence():
    # Примеры из условия
    assert multiplicative_persistence(PositiveIntegerInput(39)) == 3
    assert multiplicative_persistence(PositiveIntegerInput(4)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(999)) == 4
    # Другие
    assert multiplicative_persistence(PositiveIntegerInput(25)) == 2   # 2*5=10, 1*0=0
    assert multiplicative_persistence(PositiveIntegerInput(77)) == 4
    # Граничные
    assert multiplicative_persistence(PositiveIntegerInput(1)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(9)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(10)) == 1   # 1*0=0
    assert multiplicative_persistence(PositiveIntegerInput(99)) == 2
    print("multiplicative_persistence: OK")


def test_mse():
    # Обычные
    assert abs(mse(VectorPairInput([1, 2, 3], [1, 2, 4])) - 1/3) < 1e-9
    assert mse(VectorPairInput([0, 0], [0, 0])) == 0.0
    assert mse(VectorPairInput([5], [5])) == 0.0
    assert mse(VectorPairInput([1, 2], [3, 4])) == 4.0
    # Граничные
    assert mse(VectorPairInput([], [])) == 0.0                # пустые векторы
    assert mse(VectorPairInput([10], [0])) == 100.0
    print("mse: OK")


def test_prime_factorization():
    # Пример из условия
    assert prime_factorization(PositiveIntegerInput(86240)) == "(2**5)(5)(7**2)(11)"
    # Обычные
    assert prime_factorization(PositiveIntegerInput(12)) == "(2**2)(3)"
    assert prime_factorization(PositiveIntegerInput(7)) == "(7)"
    assert prime_factorization(PositiveIntegerInput(100)) == "(2**2)(5**2)"
    assert prime_factorization(PositiveIntegerInput(2)) == "(2)"
    # Граничные
    assert prime_factorization(PositiveIntegerInput(1)) == ""           # 1
    assert prime_factorization(PositiveIntegerInput(13)) == "(13)"      # простое
    assert prime_factorization(PositiveIntegerInput(16)) == "(2**4)"    # степень двойки
    assert prime_factorization(PositiveIntegerInput(30)) == "(2)(3)(5)"
    print("prime_factorization: OK")


def test_pyramid():
    # Обычные
    assert pyramid(PositiveIntegerInput(1)) == 1              # 1^2
    assert pyramid(PositiveIntegerInput(5)) == 2              # 1+4
    assert pyramid(PositiveIntegerInput(14)) == 3             # 1+4+9
    assert pyramid(PositiveIntegerInput(30)) == 4             # 1+4+9+16
    # Невозможные
    assert pyramid(PositiveIntegerInput(2)) == "It is impossible"
    assert pyramid(PositiveIntegerInput(3)) == "It is impossible"
    assert pyramid(PositiveIntegerInput(15)) == "It is impossible"
    # Граничные
    assert pyramid(PositiveIntegerInput(55)) == 5             # 1+4+9+16+25
    print("pyramid: OK")


def test_is_balanced_number():
    # Примеры из условия
    assert is_balanced_number(PositiveIntegerInput(1234006)) is True
    assert is_balanced_number(PositiveIntegerInput(123456)) is False
    # Другие
    assert is_balanced_number(PositiveIntegerInput(12321)) is True
    assert is_balanced_number(PositiveIntegerInput(11)) is True
    assert is_balanced_number(PositiveIntegerInput(12)) is True
    assert is_balanced_number(PositiveIntegerInput(123)) is False
    # Граничные
    assert is_balanced_number(PositiveIntegerInput(1)) is True
    assert is_balanced_number(PositiveIntegerInput(5)) is True
    assert is_balanced_number(PositiveIntegerInput(1001)) is True
    assert is_balanced_number(PositiveIntegerInput(1221)) is True
    assert is_balanced_number(PositiveIntegerInput(123321)) is True
    assert is_balanced_number(PositiveIntegerInput(123421)) is True
    assert is_balanced_number(PositiveIntegerInput(123451)) is False
    print("is_balanced_number: OK")


if __name__ == "__main__":
    test_count_vowels()
    test_has_unique_characters()
    test_count_one_bits()
    test_multiplicative_persistence()
    test_mse()
    test_prime_factorization()
    test_pyramid()
    test_is_balanced_number()
    print("\nВсе тесты пройдены успешно")