"""Copy-ready NumPy and pandas practice examples."""

import numpy as np
import pandas as pd


def print_title(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def numpy_examples():
    print_title("NUMPY OPERATIONS")

    np.random.seed(42)

    arr1 = np.array([10, 20, 30, 40, 50])
    arr2 = np.array([1, 2, 3, 4, 5])
    matrix = np.array([[1, 2, 3], [4, 5, 6]])

    print("Create arrays:")
    print("arr1 =", arr1)
    print("arr2 =", arr2)
    print("matrix =\n", matrix)

    print("\nArray properties:")
    print("shape =", matrix.shape)
    print("ndim =", matrix.ndim)
    print("size =", matrix.size)
    print("dtype =", matrix.dtype)

    print("\nSpecial arrays:")
    print("zeros =\n", np.zeros((2, 3)))
    print("ones =\n", np.ones((2, 2)))
    print("identity =\n", np.eye(3))
    print("arange =", np.arange(0, 10, 2))
    print("linspace =", np.linspace(0, 1, 5))
    print("random integers =\n", np.random.randint(1, 20, size=(3, 3)))

    print("\nArithmetic operations:")
    print("arr1 + arr2 =", arr1 + arr2)
    print("arr1 - arr2 =", arr1 - arr2)
    print("arr1 * arr2 =", arr1 * arr2)
    print("arr1 / arr2 =", arr1 / arr2)
    print("arr2 ** 2 =", arr2**2)
    print("sqrt(arr1) =", np.sqrt(arr1))

    print("\nIndexing and slicing:")
    print("first element =", arr1[0])
    print("last element =", arr1[-1])
    print("slice [1:4] =", arr1[1:4])
    print("matrix first row =", matrix[0])
    print("matrix second column =", matrix[:, 1])

    print("\nReshape, flatten, transpose:")
    reshaped = np.arange(1, 13).reshape(3, 4)
    print("reshaped =\n", reshaped)
    print("flatten =", reshaped.flatten())
    print("transpose =\n", reshaped.T)

    print("\nAggregate functions:")
    print("sum =", np.sum(arr1))
    print("mean =", np.mean(arr1))
    print("median =", np.median(arr1))
    print("std =", np.std(arr1))
    print("min =", np.min(arr1))
    print("max =", np.max(arr1))
    print("column-wise sum =", np.sum(reshaped, axis=0))
    print("row-wise sum =", np.sum(reshaped, axis=1))

    print("\nBoolean filtering:")
    print("arr1 > 25 =", arr1 > 25)
    print("filtered values =", arr1[arr1 > 25])
    print("np.where(arr1 > 25, 'High', 'Low') =", np.where(arr1 > 25, "High", "Low"))

    print("\nSorting and unique values:")
    sample = np.array([4, 8, 2, 8, 1, 4, 9])
    print("sample =", sample)
    print("sorted =", np.sort(sample))
    print("unique =", np.unique(sample))

    print("\nConcatenate and stack:")
    print("concatenate =", np.concatenate([arr1, arr2]))
    print("vstack =\n", np.vstack([arr1, arr2]))
    print("hstack =", np.hstack([arr1, arr2]))

    print("\nBroadcasting:")
    print("matrix + 10 =\n", matrix + 10)
    print("matrix * [1, 10, 100] =\n", matrix * np.array([1, 10, 100]))

    print("\nLinear algebra:")
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    print("dot product =\n", np.dot(a, b))
    print("determinant of a =", np.linalg.det(a))
    print("inverse of a =\n", np.linalg.inv(a))
