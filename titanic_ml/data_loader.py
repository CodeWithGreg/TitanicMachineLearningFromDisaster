"""Data loading and schema validation for Titanic datasets."""

import os

import pandas as pd

EXPECTED_COLUMNS = {
    "PassengerId", "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
}


def load_data(filepath: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Args:
        filepath: Path to the CSV file.

    Returns:
        A pandas DataFrame containing the CSV data.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)


def validate_schema(df: pd.DataFrame, is_training: bool = False) -> None:
    """Validate that a DataFrame contains the expected Titanic columns.

    Args:
        df: The DataFrame to validate.
        is_training: If True, also checks for the 'Survived' column.

    Raises:
        ValueError: If expected columns are missing, with a message listing them.
    """
    required = set(EXPECTED_COLUMNS)
    if is_training:
        required = required | {"Survived"}

    missing = required - set(df.columns)
    if missing:
        sorted_missing = sorted(missing)
        raise ValueError(f"Missing columns: {sorted_missing}")
