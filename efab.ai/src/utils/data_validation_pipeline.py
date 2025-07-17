"""
Automated Data Validation Pipeline
Comprehensive data validation system for the Beverly Knits AI Supply Chain Planner.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, date
from pathlib import Path
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import re

from .error_handling import handle_errors, ErrorCategory, ErrorSeverity, global_error_handler

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ValidationType(Enum):
    """Types of validation checks."""
    SCHEMA = "schema"
    DATA_TYPE = "data_type"
    RANGE = "range"
    FORMAT = "format"
    BUSINESS_RULE = "business_rule"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    UNIQUENESS = "uniqueness"

@dataclass
class ValidationRule:
    """Data validation rule definition."""
    rule_id: str
    rule_type: ValidationType
    column: str
    condition: str
    expected_value: Any
    severity: ValidationSeverity
    description: str
    auto_fix: bool = False
    fix_action: Optional[str] = None

@dataclass
class ValidationIssue:
    """Data validation issue report."""
    rule_id: str
    severity: ValidationSeverity
    issue_type: ValidationType
    column: str
    row_index: Optional[int]
    current_value: Any
    expected_value: Any
    description: str
    auto_fixable: bool
    fix_applied: bool = False

@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    dataset_name: str
    validation_timestamp: str
    total_rows: int
    total_columns: int
    rules_executed: int
    issues_found: List[ValidationIssue]
    summary: Dict[str, int]
    data_quality_score: float
    recommendations: List[str]

class DataValidationPipeline:
    """Automated data validation pipeline for Beverly Knits data."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/validation_rules.json"
        self.validation_rules = {}
        self.auto_fix_enabled = True
        self.validation_history = []
        
        # Load validation rules
        self._load_validation_rules()
        
        # Setup validation schemas
        self._setup_validation_schemas()
    
    def _load_validation_rules(self):
        """Load validation rules from configuration."""
        
        # Default validation rules for Beverly Knits data
        default_rules = {
            "yarn_materials": [
                ValidationRule(
                    rule_id="yarn_id_required",
                    rule_type=ValidationType.COMPLETENESS,
                    column="Yarn_ID",
                    condition="not_null",
                    expected_value=None,
                    severity=ValidationSeverity.CRITICAL,
                    description="Yarn ID is required for all materials"
                ),
                ValidationRule(
                    rule_id="cost_positive",
                    rule_type=ValidationType.RANGE,
                    column="Cost_Pound",
                    condition="greater_than",
                    expected_value=0,
                    severity=ValidationSeverity.ERROR,
                    description="Cost per pound must be positive",
                    auto_fix=True,
                    fix_action="set_default_cost"
                ),
                ValidationRule(
                    rule_id="inventory_non_negative",
                    rule_type=ValidationType.RANGE,
                    column="Inventory",
                    condition="greater_equal",
                    expected_value=0,
                    severity=ValidationSeverity.WARNING,
                    description="Inventory should not be negative",
                    auto_fix=True,
                    fix_action="set_zero"
                )
            ],
            "suppliers": [
                ValidationRule(
                    rule_id="supplier_id_required",
                    rule_type=ValidationType.COMPLETENESS,
                    column="Supplier_ID",
                    condition="not_null",
                    expected_value=None,
                    severity=ValidationSeverity.CRITICAL,
                    description="Supplier ID is required"
                ),
                ValidationRule(
                    rule_id="lead_time_reasonable",
                    rule_type=ValidationType.RANGE,
                    column="Lead_Time_Days",
                    condition="between",
                    expected_value=(1, 365),
                    severity=ValidationSeverity.WARNING,
                    description="Lead time should be between 1 and 365 days"
                )
            ],
            "boms": [
                ValidationRule(
                    rule_id="style_id_required",
                    rule_type=ValidationType.COMPLETENESS,
                    column="Style_ID",
                    condition="not_null",
                    expected_value=None,
                    severity=ValidationSeverity.CRITICAL,
                    description="Style ID is required for BOMs"
                ),
                ValidationRule(
                    rule_id="bom_percentage_valid",
                    rule_type=ValidationType.RANGE,
                    column="BOM_Percentage",
                    condition="between",
                    expected_value=(0, 1),
                    severity=ValidationSeverity.ERROR,
                    description="BOM percentage must be between 0 and 1",
                    auto_fix=True,
                    fix_action="normalize_percentage"
                )
            ],
            "sales_orders": [
                ValidationRule(
                    rule_id="order_quantity_positive",
                    rule_type=ValidationType.RANGE,
                    column="Quantity",
                    condition="greater_than",
                    expected_value=0,
                    severity=ValidationSeverity.ERROR,
                    description="Order quantity must be positive"
                ),
                ValidationRule(
                    rule_id="ship_date_format",
                    rule_type=ValidationType.FORMAT,
                    column="Ship_Date",
                    condition="date_format",
                    expected_value="%Y-%m-%d",
                    severity=ValidationSeverity.WARNING,
                    description="Ship date should be in YYYY-MM-DD format"
                )
            ]
        }
        
        self.validation_rules = default_rules
        
        # Try to load from config file if it exists
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    custom_rules = json.load(f)
                    # Merge custom rules with defaults
                    for dataset, rules in custom_rules.items():
                        if dataset in self.validation_rules:
                            self.validation_rules[dataset].extend([
                                ValidationRule(**rule) for rule in rules
                            ])
        except Exception as e:
            logger.warning(f"Could not load custom validation rules: {e}")
    
    def _setup_validation_schemas(self):
        """Setup expected schemas for different datasets."""
        
        self.expected_schemas = {
            "yarn_materials": {
                "required_columns": ["Yarn_ID", "Supplier", "Description"],
                "optional_columns": ["Cost_Pound", "Inventory", "Type", "Color"],
                "data_types": {
                    "Yarn_ID": ["object", "string"],
                    "Cost_Pound": ["float64", "int64"],
                    "Inventory": ["float64", "int64"]
                }
            },
            "suppliers": {
                "required_columns": ["Supplier_ID"],
                "optional_columns": ["Lead_Time_Days", "MOQ", "Reliability_Score"],
                "data_types": {
                    "Supplier_ID": ["object", "string"],
                    "Lead_Time_Days": ["int64", "float64"],
                    "MOQ": ["float64", "int64"]
                }
            },
            "boms": {
                "required_columns": ["Style_ID", "Yarn_ID", "BOM_Percentage"],
                "optional_columns": ["unit"],
                "data_types": {
                    "Style_ID": ["object", "string"],
                    "Yarn_ID": ["object", "string"],
                    "BOM_Percentage": ["float64"]
                }
            },
            "sales_orders": {
                "required_columns": ["Style_ID", "Quantity"],
                "optional_columns": ["Ship_Date", "Order_Date", "Customer"],
                "data_types": {
                    "Style_ID": ["object", "string"],
                    "Quantity": ["int64", "float64"]
                }
            }
        }
    
    @handle_errors(category=ErrorCategory.DATA_VALIDATION)
    def validate_dataset(
        self, 
        df: pd.DataFrame, 
        dataset_type: str,
        dataset_name: Optional[str] = None
    ) -> ValidationReport:
        """Validate a dataset according to predefined rules."""
        
        dataset_name = dataset_name or f"{dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting validation for dataset: {dataset_name}")
        
        issues = []
        rules_executed = 0
        
        # Get validation rules for this dataset type
        rules = self.validation_rules.get(dataset_type, [])
        
        # Schema validation
        schema_issues = self._validate_schema(df, dataset_type)
        issues.extend(schema_issues)
        
        # Apply validation rules
        for rule in rules:
            try:
                rule_issues = self._apply_validation_rule(df, rule)
                issues.extend(rule_issues)
                rules_executed += 1
            except Exception as e:
                logger.error(f"Failed to apply validation rule {rule.rule_id}: {e}")
        
        # Calculate data quality score
        quality_score = self._calculate_quality_score(df, issues)
        
        # Generate summary
        summary = self._generate_validation_summary(issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, df)
        
        # Create validation report
        report = ValidationReport(
            dataset_name=dataset_name,
            validation_timestamp=datetime.now().isoformat(),
            total_rows=len(df),
            total_columns=len(df.columns),
            rules_executed=rules_executed,
            issues_found=issues,
            summary=summary,
            data_quality_score=quality_score,
            recommendations=recommendations
        )
        
        # Store validation history
        self.validation_history.append(report)
        
        logger.info(f"Validation completed: {len(issues)} issues found, quality score: {quality_score:.2f}")
        
        return report
    
    def _validate_schema(self, df: pd.DataFrame, dataset_type: str) -> List[ValidationIssue]:
        """Validate dataset schema."""
        
        issues = []
        schema = self.expected_schemas.get(dataset_type, {})
        
        if not schema:
            return issues
        
        # Check required columns
        required_columns = schema.get("required_columns", [])
        missing_columns = set(required_columns) - set(df.columns)
        
        for column in missing_columns:
            issues.append(ValidationIssue(
                rule_id="missing_required_column",
                severity=ValidationSeverity.CRITICAL,
                issue_type=ValidationType.SCHEMA,
                column=column,
                row_index=None,
                current_value=None,
                expected_value="column_present",
                description=f"Required column '{column}' is missing",
                auto_fixable=False
            ))
        
        # Check data types
        expected_types = schema.get("data_types", {})
        for column, expected_type_list in expected_types.items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                if actual_type not in expected_type_list:
                    issues.append(ValidationIssue(
                        rule_id="incorrect_data_type",
                        severity=ValidationSeverity.WARNING,
                        issue_type=ValidationType.DATA_TYPE,
                        column=column,
                        row_index=None,
                        current_value=actual_type,
                        expected_value=expected_type_list[0],
                        description=f"Column '{column}' has type '{actual_type}', expected one of {expected_type_list}",
                        auto_fixable=True
                    ))
        
        return issues
    
    def _apply_validation_rule(self, df: pd.DataFrame, rule: ValidationRule) -> List[ValidationIssue]:
        """Apply a single validation rule to the dataset."""
        
        issues = []
        
        if rule.column not in df.columns:
            return issues
        
        column_data = df[rule.column]
        
        # Apply different validation conditions
        if rule.condition == "not_null":
            mask = column_data.isnull()
            
        elif rule.condition == "greater_than":
            mask = column_data <= rule.expected_value
            
        elif rule.condition == "greater_equal":
            mask = column_data < rule.expected_value
            
        elif rule.condition == "less_than":
            mask = column_data >= rule.expected_value
            
        elif rule.condition == "between":
            min_val, max_val = rule.expected_value
            mask = (column_data < min_val) | (column_data > max_val)
            
        elif rule.condition == "in_list":
            mask = ~column_data.isin(rule.expected_value)
            
        elif rule.condition == "regex_match":
            pattern = rule.expected_value
            mask = ~column_data.astype(str).str.match(pattern, na=False)
            
        elif rule.condition == "date_format":
            try:
                pd.to_datetime(column_data, format=rule.expected_value, errors='coerce')
                mask = pd.to_datetime(column_data, format=rule.expected_value, errors='coerce').isnull()
            except:
                mask = pd.Series([True] * len(column_data))
                
        else:
            logger.warning(f"Unknown validation condition: {rule.condition}")
            return issues
        
        # Create issues for violations
        violation_indices = df[mask].index
        
        for idx in violation_indices:
            current_value = column_data.iloc[idx] if idx < len(column_data) else None
            
            issues.append(ValidationIssue(
                rule_id=rule.rule_id,
                severity=rule.severity,
                issue_type=rule.rule_type,
                column=rule.column,
                row_index=idx,
                current_value=current_value,
                expected_value=rule.expected_value,
                description=rule.description,
                auto_fixable=rule.auto_fix
            ))
        
        return issues
    
    def _calculate_quality_score(self, df: pd.DataFrame, issues: List[ValidationIssue]) -> float:
        """Calculate overall data quality score (0-100)."""
        
        if len(df) == 0:
            return 0.0
        
        # Weight issues by severity
        severity_weights = {
            ValidationSeverity.CRITICAL: 10,
            ValidationSeverity.ERROR: 5,
            ValidationSeverity.WARNING: 2,
            ValidationSeverity.INFO: 1
        }
        
        total_penalty = sum(severity_weights.get(issue.severity, 1) for issue in issues)
        max_possible_penalty = len(df) * len(df.columns) * severity_weights[ValidationSeverity.CRITICAL]
        
        if max_possible_penalty == 0:
            return 100.0
        
        penalty_ratio = min(1.0, total_penalty / max_possible_penalty)
        quality_score = (1.0 - penalty_ratio) * 100
        
        return max(0.0, quality_score)
    
    def _generate_validation_summary(self, issues: List[ValidationIssue]) -> Dict[str, int]:
        """Generate validation summary statistics."""
        
        summary = {
            "total_issues": len(issues),
            "critical_issues": 0,
            "error_issues": 0,
            "warning_issues": 0,
            "info_issues": 0,
            "auto_fixable_issues": 0
        }
        
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                summary["critical_issues"] += 1
            elif issue.severity == ValidationSeverity.ERROR:
                summary["error_issues"] += 1
            elif issue.severity == ValidationSeverity.WARNING:
                summary["warning_issues"] += 1
            elif issue.severity == ValidationSeverity.INFO:
                summary["info_issues"] += 1
            
            if issue.auto_fixable:
                summary["auto_fixable_issues"] += 1
        
        return summary
    
    def _generate_recommendations(self, issues: List[ValidationIssue], df: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations based on validation issues."""
        
        recommendations = []
        
        # Group issues by type
        issue_counts = {}
        for issue in issues:
            issue_type = f"{issue.issue_type.value}_{issue.severity.value}"
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        # Generate specific recommendations
        if issue_counts.get("completeness_critical", 0) > 0:
            recommendations.append("🔴 CRITICAL: Address missing required data fields immediately")
        
        if issue_counts.get("range_error", 0) > 0:
            recommendations.append("🟡 ERROR: Review and correct out-of-range values")
        
        if issue_counts.get("data_type_warning", 0) > 0:
            recommendations.append("⚪ WARNING: Convert data types to expected formats")
        
        if sum(1 for issue in issues if issue.auto_fixable) > 0:
            recommendations.append("🔧 AUTO-FIX: Enable automatic fixes for correctable issues")
        
        # Data-specific recommendations
        if len(df) == 0:
            recommendations.append("📊 DATA: Dataset is empty - verify data source")
        
        completeness = (df.count().sum() / (len(df) * len(df.columns))) * 100 if len(df) > 0 else 0
        if completeness < 80:
            recommendations.append(f"📈 COMPLETENESS: Data is {completeness:.1f}% complete - investigate missing values")
        
        return recommendations
    
    @handle_errors(category=ErrorCategory.DATA_VALIDATION)
    def auto_fix_issues(self, df: pd.DataFrame, issues: List[ValidationIssue]) -> Tuple[pd.DataFrame, List[ValidationIssue]]:
        """Automatically fix correctable validation issues."""
        
        if not self.auto_fix_enabled:
            logger.info("Auto-fix is disabled")
            return df, issues
        
        fixed_df = df.copy()
        fixed_issues = []
        
        for issue in issues:
            if not issue.auto_fixable:
                continue
            
            try:
                # Apply fixes based on fix action
                if hasattr(self, f"_fix_{issue.rule_id}"):
                    fix_method = getattr(self, f"_fix_{issue.rule_id}")
                    fixed_df = fix_method(fixed_df, issue)
                    issue.fix_applied = True
                    fixed_issues.append(issue)
                else:
                    # Generic fixes
                    fixed_df = self._apply_generic_fix(fixed_df, issue)
                    issue.fix_applied = True
                    fixed_issues.append(issue)
                    
            except Exception as e:
                logger.error(f"Failed to apply auto-fix for {issue.rule_id}: {e}")
        
        logger.info(f"Auto-fixed {len(fixed_issues)} issues")
        return fixed_df, fixed_issues
    
    def _apply_generic_fix(self, df: pd.DataFrame, issue: ValidationIssue) -> pd.DataFrame:
        """Apply generic fixes for common issues."""
        
        if issue.rule_id == "cost_positive" and issue.row_index is not None:
            # Set default cost for materials with zero/negative cost
            df.loc[issue.row_index, issue.column] = 10.0  # Default cost
            
        elif issue.rule_id == "inventory_non_negative" and issue.row_index is not None:
            # Set negative inventory to zero
            df.loc[issue.row_index, issue.column] = 0.0
            
        elif issue.rule_id == "bom_percentage_valid" and issue.row_index is not None:
            # Normalize BOM percentages
            current_value = df.loc[issue.row_index, issue.column]
            if current_value > 1:
                df.loc[issue.row_index, issue.column] = current_value / 100
            elif current_value < 0:
                df.loc[issue.row_index, issue.column] = 0.0
        
        return df
    
    def generate_validation_report_file(self, report: ValidationReport, output_path: Optional[str] = None) -> str:
        """Generate a detailed validation report file."""
        
        if output_path is None:
            output_path = f"logs/validation_report_{report.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        report_content = f"""
DATA VALIDATION REPORT
{'='*60}

Dataset: {report.dataset_name}
Validation Date: {report.validation_timestamp}
Total Rows: {report.total_rows:,}
Total Columns: {report.total_columns}
Rules Executed: {report.rules_executed}
Data Quality Score: {report.data_quality_score:.2f}/100

SUMMARY
{'='*60}
Total Issues: {report.summary['total_issues']}
Critical Issues: {report.summary['critical_issues']}
Error Issues: {report.summary['error_issues']}
Warning Issues: {report.summary['warning_issues']}
Info Issues: {report.summary['info_issues']}
Auto-fixable Issues: {report.summary['auto_fixable_issues']}

RECOMMENDATIONS
{'='*60}
"""
        
        for i, recommendation in enumerate(report.recommendations, 1):
            report_content += f"{i}. {recommendation}\n"
        
        if report.issues_found:
            report_content += f"\nDETAILED ISSUES\n{'='*60}\n"
            
            for issue in report.issues_found[:50]:  # Limit to first 50 issues
                report_content += f"""
Issue ID: {issue.rule_id}
Severity: {issue.severity.value.upper()}
Type: {issue.issue_type.value}
Column: {issue.column}
Row: {issue.row_index if issue.row_index is not None else 'N/A'}
Current Value: {issue.current_value}
Expected: {issue.expected_value}
Description: {issue.description}
Auto-fixable: {'Yes' if issue.auto_fixable else 'No'}
{'-'*40}
"""
        
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Validation report saved to: {output_path}")
        return output_path
    
    def validate_all_datasets(self, data_path: str = "data/live/") -> Dict[str, ValidationReport]:
        """Validate all datasets in the specified directory."""
        
        logger.info(f"Starting validation for all datasets in {data_path}")
        
        results = {}
        data_dir = Path(data_path)
        
        # Dataset type mapping based on filename patterns
        dataset_mapping = {
            "yarn": "yarn_materials",
            "supplier": "suppliers",
            "bom": "boms",
            "sales": "sales_orders",
            "inventory": "yarn_materials"
        }
        
        for csv_file in data_dir.glob("*.csv"):
            try:
                # Determine dataset type from filename
                filename_lower = csv_file.name.lower()
                dataset_type = "unknown"
                
                for pattern, dtype in dataset_mapping.items():
                    if pattern in filename_lower:
                        dataset_type = dtype
                        break
                
                # Load and validate dataset
                df = pd.read_csv(csv_file, encoding='utf-8-sig')
                report = self.validate_dataset(df, dataset_type, csv_file.stem)
                results[csv_file.name] = report
                
                # Generate individual report file
                self.generate_validation_report_file(report)
                
            except Exception as e:
                logger.error(f"Failed to validate {csv_file.name}: {e}")
        
        logger.info(f"Completed validation for {len(results)} datasets")
        return results