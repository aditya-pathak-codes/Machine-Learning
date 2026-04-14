"""Copy-ready NumPy and pandas practice examples."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def pandas_examples():
    print_title("PANDAS OPERATIONS")

    data = {
        "Name": ["Amit", "Sara", "John", "Priya", "David", "Neha"],
        "Age": [23, 28, 35, 25, 32, 29],
        "City": ["Delhi", "Mumbai", "Bangalore", "Delhi", "Chennai", "Mumbai"],
        "Department": ["IT", "HR", "Finance", "IT", "HR", "Finance"],
        "Salary": [50000, 60000, 75000, 52000, 61000, 72000],
        "Score": [88, 92, np.nan, 85, 90, 95],
    }

    df = pd.DataFrame(data)

    print("Create DataFrame:")
    print(df)

    print("\nBasic inspection:")
    print("head =\n", df.head())
    print("tail =\n", df.tail(2))
    print("shape =", df.shape)
    print("columns =", list(df.columns))
    print("dtypes =\n", df.dtypes)
    print("describe =\n", df.describe(numeric_only=True))

    print("\nSelect columns and rows:")
    print("Single column =\n", df["Name"])
    print("Multiple columns =\n", df[["Name", "Salary"]])
    print("loc row 0 =\n", df.loc[0])
    print("iloc first 3 rows =\n", df.iloc[:3])

    print("\nFiltering rows:")
    print("Age > 28 =\n", df[df["Age"] > 28])
    print("City is Delhi and Salary > 50000 =\n", df[(df["City"] == "Delhi") & (df["Salary"] > 50000)])

    print("\nAdd new columns:")
    df["Bonus"] = df["Salary"] * 0.10
    df["Tax"] = df["Salary"] * 0.05
    df["NetSalary"] = df["Salary"] + df["Bonus"] - df["Tax"]
    df["SalaryBand"] = df["Salary"].apply(lambda value: "High" if value >= 60000 else "Medium")
    print(df)

    print("\nUpdate values with loc:")
    df.loc[df["Department"] == "IT", "Salary"] = df.loc[df["Department"] == "IT", "Salary"] + 3000
    print(df[["Name", "Department", "Salary"]])

    print("\nMissing value handling:")
    print("missing values =\n", df.isnull().sum())
    df["Score"] = df["Score"].fillna(df["Score"].mean())
    print("Score after fillna =\n", df["Score"])

    print("\nSorting:")
    print("Sort by Salary descending =\n", df.sort_values(by="Salary", ascending=False))
    print("Sort by City then Age =\n", df.sort_values(by=["City", "Age"]))

    print("\nGroupBy operations:")
    grouped = df.groupby("Department").agg(
        AverageSalary=("Salary", "mean"),
        MaxSalary=("Salary", "max"),
        AverageScore=("Score", "mean"),
    )
    print(grouped)

    print("\nValue counts:")
    print(df["City"].value_counts())

    print("\nString operations:")
    print(df["Name"].str.upper())

    print("\nRename and drop columns:")
    renamed_df = df.rename(columns={"NetSalary": "TakeHome"})
    dropped_df = renamed_df.drop(columns=["Tax"])
    print(dropped_df.head())

    print("\nSet index and reset index:")
    indexed_df = df.set_index("Name")
    print(indexed_df)
    print("\nReset index =\n", indexed_df.reset_index())

    print("\nDate and time operations:")
    df["JoinDate"] = pd.to_datetime(
        ["2022-01-10", "2021-06-15", "2020-03-20", "2023-02-05", "2021-11-11", "2022-08-01"]
    )
    print(df[["Name", "JoinDate"]])
    print("Join year =\n", df["JoinDate"].dt.year)
    print("Join month name =\n", df["JoinDate"].dt.month_name())

    print("\nPivot table:")
    pivot = pd.pivot_table(df, values="Salary", index="Department", columns="City", aggfunc="mean")
    print(pivot)

    print("\nMerge:")
    department_info = pd.DataFrame(
        {
            "Department": ["IT", "HR", "Finance"],
            "Manager": ["Rohan", "Meera", "Karan"],
        }
    )
    merged_df = pd.merge(df, department_info, on="Department", how="left")
    print(merged_df[["Name", "Department", "Manager"]])

    print("\nConcatenate:")
    new_rows = pd.DataFrame(
        {
            "Name": ["Kiran"],
            "Age": [27],
            "City": ["Pune"],
            "Department": ["IT"],
            "Salary": [58000],
            "Score": [89],
            "Bonus": [5800.0],
            "Tax": [2900.0],
            "NetSalary": [60900.0],
            "SalaryBand": ["Medium"],
            "JoinDate": [pd.Timestamp("2024-01-12")],
        }
    )
    combined_df = pd.concat([df, new_rows], ignore_index=True)
    print(combined_df.tail())

    print("\nCorrelation between numeric columns:")
    print(df.corr(numeric_only=True))


def matplotlib_examples():
    print_title("MATPLOTLIB OPERATIONS")

    output_dir = Path(__file__).with_name("plots")
    output_dir.mkdir(exist_ok=True)

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    sales = np.array([12, 15, 14, 18, 22, 24])
    expenses = np.array([8, 9, 10, 11, 13, 14])
    study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    scores = np.array([52, 55, 61, 65, 70, 74, 79, 84])
    rng = np.random.default_rng(42)
    exam_scores = rng.normal(loc=72, scale=8, size=120)

    print("Creating plots and saving them in:", output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Matplotlib Practice Charts", fontsize=14)

    axes[0, 0].plot(months, sales, marker="o", linewidth=2, color="teal", label="Sales")
    axes[0, 0].plot(months, expenses, marker="s", linewidth=2, color="orange", label="Expenses")
    axes[0, 0].set_title("Line Plot")
    axes[0, 0].set_xlabel("Month")
    axes[0, 0].set_ylabel("Amount")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    departments = ["IT", "HR", "Finance", "Sales"]
    employees = [12, 7, 6, 9]
    axes[0, 1].bar(departments, employees, color=["steelblue", "salmon", "gold", "mediumseagreen"])
    axes[0, 1].set_title("Bar Chart")
    axes[0, 1].set_xlabel("Department")
    axes[0, 1].set_ylabel("Employees")

    axes[1, 0].hist(exam_scores, bins=10, color="slateblue", edgecolor="black", alpha=0.8)
    axes[1, 0].set_title("Histogram")
    axes[1, 0].set_xlabel("Score")
    axes[1, 0].set_ylabel("Frequency")

    axes[1, 1].scatter(study_hours, scores, color="crimson", s=80)
    axes[1, 1].set_title("Scatter Plot")
    axes[1, 1].set_xlabel("Study Hours")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_path = output_dir / "matplotlib_practice.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print("Saved practice chart to:", output_path)


