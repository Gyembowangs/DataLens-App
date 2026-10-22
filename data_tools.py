
import json

import pandas as pd
from langchain.tools import tool


def create_data_tools(df: pd.DataFrame):
    """Create predefined analysis tools for a CSV DataFrame."""

    data = df.copy()

    # --------------------------------------------------
    # Tool 1: Inspect the dataset
    # --------------------------------------------------
    @tool
    def inspect_dataset() -> str:
        """Show row count, column names, data types,
        and missing-value counts."""
        result = {
            "rows": len(data),
            "columns": list(data.columns),
            "data_types": {
                col: str(dtype)
                for col, dtype in data.dtypes.items()
            },
            "missing_values": {
                col: int(count)
                for col, count in data.isna().sum().items()
            },
        }

        return json.dumps(result, indent=2, allow_nan=False)

    # --------------------------------------------------
    # Tool 2: Summarize a column
    # --------------------------------------------------
    @tool
    def summarize_column(column: str) -> str:
        """Calculate descriptive statistics for a column.
        Input: the exact column name."""
        if column not in data.columns:
            return (
                f"Error: column '{column}' does not exist. "
                f"Available columns: {list(data.columns)}"
            )

        series = data[column]

        if pd.api.types.is_numeric_dtype(series):
            values = series.dropna()

            if values.empty:
                return (
                    f"Column '{column}' has no valid numeric values."
                )

            result = {
                "column": column,
                "count": int(values.count()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "missing": int(series.isna().sum()),
            }

        else:
            modes = series.mode(dropna=True)

            result = {
                "column": column,
                "count": int(series.count()),
                "unique_values": int(series.nunique()),
                "most_common": (
                    str(modes.iloc[0])
                    if not modes.empty
                    else None
                ),
                "missing": int(series.isna().sum()),
            }

        return json.dumps(result, indent=2, allow_nan=False)

    # --------------------------------------------------
    # Tool 3: Group and aggregate
    # --------------------------------------------------
    @tool
    def group_and_aggregate(
        group_column: str,
        value_column: str,
        operation: str = "sum",
    ) -> str:
        """Group rows by a column and aggregate another column.
        Allowed operations: sum, mean, count, min, max."""
        if group_column not in data.columns:
            return f"Error: '{group_column}' is not a valid column."

        if value_column not in data.columns:
            return f"Error: '{value_column}' is not a valid column."

        operation = operation.strip().lower()

        if operation not in {"sum", "mean", "count", "min", "max"}:
            return (
                "Error: choose sum, mean, count, min, or max."
            )

        values = data[value_column]

        if operation != "count":
            if not pd.api.types.is_numeric_dtype(values):
                return (
                    f"Error: '{value_column}' must be numeric "
                    f"for {operation}."
                )

            if values.dropna().empty:
                return (
                    f"Error: '{value_column}' has no valid values."
                )

        grouped = (
            data.groupby(group_column, dropna=False)[value_column]
            .agg(operation)
            .reset_index()
        )

        return grouped.to_json(
            orient="records",
            indent=2,
        )

    # --------------------------------------------------
    # Tool 4: Calculate revenue by group
    # --------------------------------------------------
    @tool
    def revenue_by_group(group_column: str) -> str:
        """Calculate total revenue by a category such as region
        or product. Requires quantity and unit_price columns.
        Input: the exact category column name."""
        required_columns = {
            "quantity",
            "unit_price",
            group_column,
        }

        missing_columns = required_columns - set(data.columns)

        if missing_columns:
            return (
                "Error: missing required columns: "
                f"{sorted(missing_columns)}"
            )

        quantity = pd.to_numeric(
            data["quantity"], errors="coerce"
        )
        unit_price = pd.to_numeric(
            data["unit_price"], errors="coerce"
        )

        valid = (
            quantity.notna()
            & unit_price.notna()
            & quantity.map(lambda x: pd.notna(x) and abs(x) != float("inf"))
            & unit_price.map(lambda x: pd.notna(x) and abs(x) != float("inf"))
        )

        if not valid.any():
            return (
                "Error: no valid quantity and unit_price values found."
            )

        revenue = quantity[valid] * unit_price[valid]

        # Exclude overflow or invalid results.
        finite_revenue = revenue.map(
            lambda value: pd.notna(value) and abs(value) != float("inf")
        )

        revenue = revenue[finite_revenue]
        groups = data.loc[revenue.index, group_column]

        result = (
            revenue.groupby(groups, dropna=False)
            .sum()
            .sort_values(ascending=False)
            .rename("total_revenue")
            .reset_index()
        )

        return result.to_json(
            orient="records",
            indent=2,
        )

    # --------------------------------------------------
    # Tool 5: Calculate revenue by month
    # --------------------------------------------------
    @tool
    def monthly_revenue(date_column: str) -> str:
        """Calculate total revenue for each calendar month.
        Requires a date column and numeric quantity and
        unit_price columns.
        Input: the exact name of the date column."""
        required_columns = {
            date_column,
            "quantity",
            "unit_price",
        }

        missing_columns = required_columns - set(data.columns)

        if missing_columns:
            return (
                "Error: missing required columns: "
                f"{sorted(missing_columns)}"
            )

        dates = pd.to_datetime(
            data[date_column], errors="coerce"
        )

        quantity = pd.to_numeric(
            data["quantity"], errors="coerce"
        )

        unit_price = pd.to_numeric(
            data["unit_price"], errors="coerce"
        )

        valid = (
            dates.notna()
            & quantity.notna()
            & unit_price.notna()
            & quantity.map(lambda x: abs(x) != float("inf"))
            & unit_price.map(lambda x: abs(x) != float("inf"))
        )

        if not valid.any():
            return (
                "Error: no rows contain valid dates, quantities, "
                "and unit prices."
            )

        monthly_data = pd.DataFrame({
            "month": dates.loc[valid].dt.strftime("%Y-%m"),
            "revenue": (
                quantity.loc[valid] * unit_price.loc[valid]
            ),
        })

        # Remove invalid or infinite revenue values.
        monthly_data = monthly_data[
            monthly_data["revenue"].map(
                lambda value: pd.notna(value)
                and abs(value) != float("inf")
            )
        ]

        if monthly_data.empty:
            return "Error: no valid revenue values could be calculated."

        result = (
            monthly_data.groupby("month")["revenue"]
            .sum()
            .sort_index()
            .rename("total_revenue")
            .reset_index()
        )

        return result.to_json(
            orient="records",
            indent=2,
        )

    # --------------------------------------------------
    # Return all tools to the LangChain agent
    # --------------------------------------------------
    return [
        inspect_dataset,
        summarize_column,
        group_and_aggregate,
        revenue_by_group,
        monthly_revenue,
    ]
