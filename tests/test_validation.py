"""Tests for the data validation module.

Covers schema definition, per-column checks (dtype, nullability, range,
allowed values), extra-column handling, and the validation report API.
"""

import pytest
import pandas as pd
import numpy as np
from src.validation import (
    ColumnSchema,
    DataSchema,
    DataValidator,
    ValidationReport,
    ValidationCheck,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """A clean DataFrame that should pass typical validation."""
    return pd.DataFrame({
        "age": [25.0, 30.0, 45.0, 22.0],
        "income": [50000.0, 60000.0, 80000.0, 35000.0],
        "category": ["A", "B", "A", "C"],
    })


@pytest.fixture
def strict_schema():
    """Schema that validates age, income, and category columns."""
    return DataSchema(
        columns=[
            ColumnSchema("age", dtype="float64", nullable=False, min_value=0, max_value=150),
            ColumnSchema("income", dtype="float64", nullable=False, min_value=0),
            ColumnSchema("category", allowed_values={"A", "B", "C"}),
        ],
        allow_extra_columns=False,
    )


# ---------------------------------------------------------------------------
# Schema Tests
# ---------------------------------------------------------------------------

class TestDataSchema:
    def test_column_names(self, strict_schema):
        assert strict_schema.column_names() == ["age", "income", "category"]

    def test_empty_schema(self):
        schema = DataSchema()
        assert schema.column_names() == []
        assert schema.allow_extra_columns is False


# ---------------------------------------------------------------------------
# Passing Validation
# ---------------------------------------------------------------------------

class TestPassingValidation:
    def test_clean_data_passes(self, sample_df, strict_schema):
        validator = DataValidator(strict_schema)
        report = validator.validate(sample_df)
        assert report.passed
        assert len(report.failures) == 0

    def test_extra_columns_allowed(self, sample_df):
        schema = DataSchema(
            columns=[ColumnSchema("age", dtype="float64")],
            allow_extra_columns=True,
        )
        report = DataValidator(schema).validate(sample_df)
        assert report.passed


# ---------------------------------------------------------------------------
# Dtype Checks
# ---------------------------------------------------------------------------

class TestDtypeValidation:
    def test_wrong_dtype_fails(self):
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        schema = DataSchema(columns=[ColumnSchema("x", dtype="float64")])
        report = DataValidator(schema).validate(df)
        assert not report.passed
        assert any(c.check == "dtype" and not c.passed for c in report.checks)

    def test_no_dtype_constraint_passes(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        schema = DataSchema(columns=[ColumnSchema("x")])  # no dtype
        report = DataValidator(schema).validate(df)
        # Should pass dtype check (no constraint)
        dtype_checks = [c for c in report.checks if c.check == "dtype"]
        assert len(dtype_checks) == 0


# ---------------------------------------------------------------------------
# Null Checks
# ---------------------------------------------------------------------------

class TestNullValidation:
    def test_nulls_in_non_nullable_column_fails(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        schema = DataSchema(columns=[ColumnSchema("x", nullable=False)])
        report = DataValidator(schema).validate(df)
        assert not report.passed
        failures = [c for c in report.checks if c.check == "no_nulls"]
        assert len(failures) == 1
        assert "1 null" in failures[0].message

    def test_nulls_in_nullable_column_passes(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        schema = DataSchema(columns=[ColumnSchema("x", nullable=True)])
        report = DataValidator(schema).validate(df)
        assert report.passed


# ---------------------------------------------------------------------------
# Range Checks
# ---------------------------------------------------------------------------

class TestRangeValidation:
    def test_below_minimum_fails(self):
        df = pd.DataFrame({"x": [10.0, -5.0, 20.0]})
        schema = DataSchema(columns=[ColumnSchema("x", min_value=0)])
        report = DataValidator(schema).validate(df)
        assert not report.passed

    def test_above_maximum_fails(self):
        df = pd.DataFrame({"x": [10.0, 200.0, 20.0]})
        schema = DataSchema(columns=[ColumnSchema("x", max_value=100)])
        report = DataValidator(schema).validate(df)
        assert not report.passed

    def test_within_range_passes(self):
        df = pd.DataFrame({"x": [5.0, 50.0, 95.0]})
        schema = DataSchema(columns=[ColumnSchema("x", min_value=0, max_value=100)])
        report = DataValidator(schema).validate(df)
        assert report.passed

    def test_range_ignores_nulls(self):
        df = pd.DataFrame({"x": [5.0, np.nan, 95.0]})
        schema = DataSchema(columns=[
            ColumnSchema("x", nullable=True, min_value=0, max_value=100),
        ])
        report = DataValidator(schema).validate(df)
        assert report.passed


# ---------------------------------------------------------------------------
# Allowed Values
# ---------------------------------------------------------------------------

class TestAllowedValuesValidation:
    def test_unexpected_category_fails(self):
        df = pd.DataFrame({"x": ["A", "B", "Z"]})
        schema = DataSchema(columns=[
            ColumnSchema("x", allowed_values={"A", "B", "C"}),
        ])
        report = DataValidator(schema).validate(df)
        assert not report.passed
        assert "Z" in str(report.failures[0].message)

    def test_valid_categories_pass(self):
        df = pd.DataFrame({"x": ["A", "B", "C", "A"]})
        schema = DataSchema(columns=[
            ColumnSchema("x", allowed_values={"A", "B", "C"}),
        ])
        report = DataValidator(schema).validate(df)
        assert report.passed


# ---------------------------------------------------------------------------
# Missing & Extra Columns
# ---------------------------------------------------------------------------

class TestColumnPresence:
    def test_missing_column_fails(self):
        df = pd.DataFrame({"a": [1]})
        schema = DataSchema(columns=[ColumnSchema("a"), ColumnSchema("b")])
        report = DataValidator(schema).validate(df)
        assert not report.passed
        assert any(c.check == "column_exists" for c in report.failures)

    def test_extra_column_strict_fails(self):
        df = pd.DataFrame({"a": [1], "extra": [2]})
        schema = DataSchema(
            columns=[ColumnSchema("a")],
            allow_extra_columns=False,
        )
        report = DataValidator(schema).validate(df)
        assert not report.passed
        assert any(c.check == "no_extra_columns" for c in report.failures)


# ---------------------------------------------------------------------------
# ValidationReport API
# ---------------------------------------------------------------------------

class TestValidationReport:
    def test_summary_structure(self):
        report = ValidationReport(checks=[
            ValidationCheck("col_a", "dtype", True, "OK"),
            ValidationCheck("col_b", "no_nulls", False, "Found nulls"),
        ])
        s = report.summary
        assert s["total_checks"] == 2
        assert s["passed"] == 1
        assert s["failed"] == 1
        assert s["status"] == "FAIL"

    def test_empty_report_passes(self):
        report = ValidationReport()
        assert report.passed
        assert report.summary["status"] == "PASS"
