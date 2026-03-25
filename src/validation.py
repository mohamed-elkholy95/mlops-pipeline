"""
Data validation for ML pipelines.

Data validation is the unsung hero of production ML.  Google's seminal
paper "Data Management Challenges in Production Machine Learning" (2015)
found that *data quality issues* — not model bugs — cause the majority of
ML system failures.  This module provides schema-based validation that
catches problems before they reach training or inference.

Key concepts:
    - **Schema**: a declarative description of what valid data looks like
      (column names, types, value ranges, nullability rules).
    - **Validation report**: a structured summary of every check, with
      pass/fail status and human-readable descriptions.
    - **Fail-fast**: pipelines should halt on validation failures rather
      than training on garbage data and producing garbage models.

In production, consider Google's TensorFlow Data Validation (TFDV) or
Great Expectations for more feature-rich validation.  The principles
here are the same — just implemented in a lightweight, educational way.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema Definition
# ---------------------------------------------------------------------------

@dataclass
class ColumnSchema:
    """Schema for a single DataFrame column.

    Attributes:
        name:        Column identifier.
        dtype:       Expected pandas dtype string (e.g., ``"float64"``, ``"int64"``,
                     ``"object"``).  ``None`` means any type is accepted.
        nullable:    Whether ``NaN`` / ``None`` values are allowed.
        min_value:   Lower bound (inclusive) for numeric columns.
        max_value:   Upper bound (inclusive) for numeric columns.
        allowed_values: Whitelist for categorical columns.  If set, every
                        non-null value must appear in this set.
    """

    name: str
    dtype: Optional[str] = None
    nullable: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[Set[Any]] = None


@dataclass
class DataSchema:
    """Full schema for a DataFrame — a collection of column schemas.

    Usage::

        schema = DataSchema(columns=[
            ColumnSchema("age", dtype="float64", nullable=False, min_value=0, max_value=150),
            ColumnSchema("income", dtype="float64", nullable=False, min_value=0),
            ColumnSchema("category", dtype="object", allowed_values={"A", "B", "C"}),
        ])
    """

    columns: List[ColumnSchema] = field(default_factory=list)
    allow_extra_columns: bool = False

    def column_names(self) -> List[str]:
        """Return the list of expected column names, preserving order."""
        return [c.name for c in self.columns]


# ---------------------------------------------------------------------------
# Validation Report
# ---------------------------------------------------------------------------

@dataclass
class ValidationCheck:
    """Result of a single validation check."""

    column: str
    check: str
    passed: bool
    message: str


@dataclass
class ValidationReport:
    """Aggregated validation report for a dataset.

    The report collects individual check results and provides convenience
    properties for querying overall status and filtering failures.
    """

    checks: List[ValidationCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """``True`` if every individual check passed."""
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> List[ValidationCheck]:
        """Return only the checks that failed."""
        return [c for c in self.checks if not c.passed]

    @property
    def summary(self) -> Dict[str, Any]:
        """Human-readable summary dict suitable for logging or JSON."""
        return {
            "total_checks": len(self.checks),
            "passed": sum(1 for c in self.checks if c.passed),
            "failed": sum(1 for c in self.checks if not c.passed),
            "status": "PASS" if self.passed else "FAIL",
            "failures": [
                {"column": f.column, "check": f.check, "message": f.message}
                for f in self.failures
            ],
        }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class DataValidator:
    """Validate a pandas DataFrame against a DataSchema.

    Validation flow:
        1. Check that all expected columns exist.
        2. Optionally reject unexpected (extra) columns.
        3. For each column, run type, nullability, range, and value checks.

    The validator is designed to be **non-destructive** — it never modifies
    the input DataFrame.  It only reads and reports.
    """

    def __init__(self, schema: DataSchema) -> None:
        self.schema = schema

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Run all validation checks and return a report.

        Returns:
            A ``ValidationReport`` containing every check result.  Callers
            should inspect ``report.passed`` to decide whether to proceed.
        """
        report = ValidationReport()

        # -- 1. Missing columns -------------------------------------------
        expected = set(self.schema.column_names())
        actual = set(df.columns)

        for col_name in expected - actual:
            report.checks.append(ValidationCheck(
                column=col_name,
                check="column_exists",
                passed=False,
                message=f"Expected column '{col_name}' is missing from the DataFrame.",
            ))

        # -- 2. Extra columns (if schema is strict) -----------------------
        if not self.schema.allow_extra_columns:
            for col_name in actual - expected:
                report.checks.append(ValidationCheck(
                    column=col_name,
                    check="no_extra_columns",
                    passed=False,
                    message=f"Unexpected column '{col_name}' not in schema.",
                ))

        # -- 3. Per-column checks -----------------------------------------
        for col_schema in self.schema.columns:
            if col_schema.name not in df.columns:
                continue  # already reported as missing above

            series = df[col_schema.name]
            self._check_dtype(series, col_schema, report)
            self._check_nulls(series, col_schema, report)
            self._check_range(series, col_schema, report)
            self._check_allowed_values(series, col_schema, report)

        if report.passed:
            logger.info("Data validation passed (%d checks).", len(report.checks))
        else:
            logger.warning(
                "Data validation FAILED: %d of %d checks failed.",
                len(report.failures),
                len(report.checks),
            )

        return report

    # -- Private check methods --------------------------------------------

    @staticmethod
    def _check_dtype(
        series: pd.Series,
        col: ColumnSchema,
        report: ValidationReport,
    ) -> None:
        """Verify that the column's dtype matches the schema."""
        if col.dtype is None:
            return  # no type constraint
        actual_dtype = str(series.dtype)
        ok = actual_dtype == col.dtype
        report.checks.append(ValidationCheck(
            column=col.name,
            check="dtype",
            passed=ok,
            message=(
                f"OK — dtype is '{actual_dtype}'."
                if ok
                else f"Expected dtype '{col.dtype}', got '{actual_dtype}'."
            ),
        ))

    @staticmethod
    def _check_nulls(
        series: pd.Series,
        col: ColumnSchema,
        report: ValidationReport,
    ) -> None:
        """Check whether null values are present when they shouldn't be."""
        null_count = int(series.isna().sum())
        ok = col.nullable or null_count == 0
        report.checks.append(ValidationCheck(
            column=col.name,
            check="no_nulls",
            passed=ok,
            message=(
                f"OK — {null_count} null(s) found (nullable={col.nullable})."
                if ok
                else f"Column is non-nullable but contains {null_count} null value(s)."
            ),
        ))

    @staticmethod
    def _check_range(
        series: pd.Series,
        col: ColumnSchema,
        report: ValidationReport,
    ) -> None:
        """Validate that numeric values fall within [min_value, max_value]."""
        if col.min_value is None and col.max_value is None:
            return  # no range constraint
        if not np.issubdtype(series.dtype, np.number):
            return  # range checks only apply to numeric columns

        non_null = series.dropna()
        if non_null.empty:
            return

        if col.min_value is not None:
            violations = int((non_null < col.min_value).sum())
            ok = violations == 0
            report.checks.append(ValidationCheck(
                column=col.name,
                check="min_value",
                passed=ok,
                message=(
                    f"OK — all values >= {col.min_value}."
                    if ok
                    else f"{violations} value(s) below minimum {col.min_value}."
                ),
            ))

        if col.max_value is not None:
            violations = int((non_null > col.max_value).sum())
            ok = violations == 0
            report.checks.append(ValidationCheck(
                column=col.name,
                check="max_value",
                passed=ok,
                message=(
                    f"OK — all values <= {col.max_value}."
                    if ok
                    else f"{violations} value(s) above maximum {col.max_value}."
                ),
            ))

    @staticmethod
    def _check_allowed_values(
        series: pd.Series,
        col: ColumnSchema,
        report: ValidationReport,
    ) -> None:
        """Verify that categorical values belong to the allowed set."""
        if col.allowed_values is None:
            return

        non_null = series.dropna()
        invalid = set(non_null.unique()) - col.allowed_values
        ok = len(invalid) == 0
        report.checks.append(ValidationCheck(
            column=col.name,
            check="allowed_values",
            passed=ok,
            message=(
                f"OK — all values in {col.allowed_values}."
                if ok
                else f"Found unexpected values: {invalid}."
            ),
        ))
