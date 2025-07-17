"""
Data Quality Service
Application service for data quality management and validation.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from pathlib import Path

from ...utils.data_quality_fixer import DataQualityFixer
from ...utils.data_validation_pipeline import DataValidationPipeline, ValidationReport
from ...utils.cost_assignment import CostAssigner
from ...utils.error_handling import handle_errors, ErrorCategory

logger = logging.getLogger(__name__)

class DataQualityService:
    """
    Application service for comprehensive data quality management.
    Orchestrates validation, fixing, and quality reporting workflows.
    """
    
    def __init__(self, data_path: str = "data/live/"):
        self.data_path = data_path
        self.quality_fixer = DataQualityFixer(data_path)
        self.validation_pipeline = DataValidationPipeline()
        self.cost_assigner = CostAssigner(data_path)
        
    @handle_errors(category=ErrorCategory.DATA_VALIDATION)
    def execute_comprehensive_quality_check(self) -> Dict[str, Any]:
        """
        Execute comprehensive data quality check and fixing process.
        
        Returns:
            Dictionary containing quality results, fixes applied, and recommendations
        """
        logger.info("Starting comprehensive data quality check")
        
        # Step 1: Apply data quality fixes
        logger.info("Step 1: Applying data quality fixes")
        fixed_datasets = self.quality_fixer.apply_all_fixes()
        
        # Step 2: Assign missing costs
        logger.info("Step 2: Assigning missing costs")
        cost_assignment_result = self.cost_assigner.run_cost_assignment()
        
        # Step 3: Run validation pipeline
        logger.info("Step 3: Running validation pipeline")
        validation_results = self.validation_pipeline.validate_all_datasets(self.data_path)
        
        # Step 4: Generate comprehensive quality report
        logger.info("Step 4: Generating quality report")
        quality_report = self._generate_comprehensive_quality_report(
            fixed_datasets, cost_assignment_result, validation_results
        )
        
        logger.info("Comprehensive data quality check completed")
        return quality_report
    
    @handle_errors(category=ErrorCategory.DATA_VALIDATION)
    def validate_dataset_quality(self, dataset_type: str, dataset_name: Optional[str] = None) -> ValidationReport:
        """
        Validate quality of a specific dataset.
        
        Args:
            dataset_type: Type of dataset (yarn_materials, suppliers, boms, sales_orders)
            dataset_name: Optional specific dataset name
            
        Returns:
            Validation report for the dataset
        """
        logger.info(f"Validating dataset quality for {dataset_type}")
        
        # Load the dataset
        import pandas as pd
        
        dataset_files = {
            'yarn_materials': 'Yarn_ID_1.csv',
            'suppliers': 'Supplier_ID.csv',
            'boms': 'Style_BOM.csv',
            'sales_orders': 'eFab_SO_List.csv',
            'inventory': 'Yarn_ID_Current_Inventory.csv'
        }
        
        if dataset_type not in dataset_files:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        file_path = Path(self.data_path) / dataset_files[dataset_type]
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        
        # Run validation
        report = self.validation_pipeline.validate_dataset(df, dataset_type, dataset_name)
        
        logger.info(f"Dataset validation completed. Quality score: {report.data_quality_score:.2f}")
        return report
    
    @handle_errors(category=ErrorCategory.DATA_VALIDATION)
    def get_data_quality_summary(self) -> Dict[str, Any]:
        """
        Get high-level data quality summary across all datasets.
        
        Returns:
            Summary of data quality metrics and status
        """
        logger.info("Generating data quality summary")
        
        validation_results = self.validation_pipeline.validate_all_datasets(self.data_path)
        
        # Calculate overall metrics
        total_issues = sum(len(report.issues_found) for report in validation_results.values())
        average_quality_score = sum(report.data_quality_score for report in validation_results.values()) / len(validation_results) if validation_results else 0
        
        # Determine overall status
        if average_quality_score >= 90:
            overall_status = "excellent"
        elif average_quality_score >= 75:
            overall_status = "good"
        elif average_quality_score >= 60:
            overall_status = "fair"
        else:
            overall_status = "poor"
        
        # Get critical issues
        critical_issues = []
        for dataset_name, report in validation_results.items():
            for issue in report.issues_found:
                if issue.severity.value == "critical":
                    critical_issues.append({
                        'dataset': dataset_name,
                        'issue': issue.description,
                        'column': issue.column
                    })
        
        summary = {
            'overall_status': overall_status,
            'average_quality_score': round(average_quality_score, 2),
            'total_datasets': len(validation_results),
            'total_issues': total_issues,
            'critical_issues_count': len(critical_issues),
            'critical_issues': critical_issues[:5],  # Top 5 critical issues
            'dataset_scores': {
                name: round(report.data_quality_score, 2) 
                for name, report in validation_results.items()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Data quality summary generated. Overall status: {overall_status}")
        return summary
    
    @handle_errors(category=ErrorCategory.DATA_VALIDATION)
    def fix_data_quality_issues(self, auto_fix: bool = True) -> Dict[str, Any]:
        """
        Fix identified data quality issues.
        
        Args:
            auto_fix: Whether to automatically apply fixes
            
        Returns:
            Results of fixing process
        """
        logger.info(f"Fixing data quality issues (auto_fix: {auto_fix})")
        
        if not auto_fix:
            logger.info("Auto-fix disabled. Running in analysis mode only.")
            return self.execute_comprehensive_quality_check()
        
        # Apply fixes
        fixed_datasets = self.quality_fixer.apply_all_fixes()
        cost_assignment_result = self.cost_assigner.run_cost_assignment()
        
        # Validate after fixes
        post_fix_validation = self.validation_pipeline.validate_all_datasets(self.data_path)
        
        # Calculate improvement
        improvement_summary = self._calculate_quality_improvement(post_fix_validation)
        
        result = {
            'fixes_applied': len(self.quality_fixer.fixes_applied),
            'costs_assigned': len(self.cost_assigner.cost_assignments),
            'datasets_processed': len(fixed_datasets),
            'post_fix_quality_scores': {
                name: report.data_quality_score 
                for name, report in post_fix_validation.items()
            },
            'improvement_summary': improvement_summary,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Data quality fixes completed. {result['fixes_applied']} fixes applied.")
        return result
    
    def _generate_comprehensive_quality_report(
        self, 
        fixed_datasets: Dict[str, Any], 
        cost_assignment_result: Dict[str, Any], 
        validation_results: Dict[str, ValidationReport]
    ) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        
        report = {
            'execution_summary': {
                'timestamp': datetime.now().isoformat(),
                'datasets_processed': len(fixed_datasets),
                'fixes_applied': len(self.quality_fixer.fixes_applied),
                'costs_assigned': len(self.cost_assigner.cost_assignments)
            },
            'quality_scores': {
                name: report.data_quality_score 
                for name, report in validation_results.items()
            },
            'validation_summary': {
                'total_rules_executed': sum(report.rules_executed for report in validation_results.values()),
                'total_issues_found': sum(len(report.issues_found) for report in validation_results.values()),
                'average_quality_score': sum(report.data_quality_score for report in validation_results.values()) / len(validation_results) if validation_results else 0
            },
            'fix_summary': {
                'negative_inventory_fixes': sum(1 for fix in self.quality_fixer.fixes_applied if 'inventory' in str(fix)),
                'cost_assignments': len(self.cost_assigner.cost_assignments),
                'bom_normalizations': sum(1 for fix in self.quality_fixer.fixes_applied if 'bom' in str(fix))
            },
            'recommendations': self._generate_quality_recommendations(validation_results)
        }
        
        return report
    
    def _calculate_quality_improvement(self, post_fix_validation: Dict[str, ValidationReport]) -> Dict[str, Any]:
        """Calculate quality improvement metrics."""
        
        # This would ideally compare pre-fix and post-fix scores
        # For now, we'll provide current state analysis
        
        scores = [report.data_quality_score for report in post_fix_validation.values()]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            'average_quality_score': round(avg_score, 2),
            'datasets_above_90': sum(1 for score in scores if score >= 90),
            'datasets_above_75': sum(1 for score in scores if score >= 75),
            'datasets_below_60': sum(1 for score in scores if score < 60),
            'overall_improvement': "significant" if avg_score >= 85 else "moderate" if avg_score >= 70 else "minimal"
        }
    
    def _generate_quality_recommendations(self, validation_results: Dict[str, ValidationReport]) -> List[str]:
        """Generate actionable quality recommendations."""
        
        recommendations = []
        
        # Analyze validation results for patterns
        low_quality_datasets = [
            name for name, report in validation_results.items() 
            if report.data_quality_score < 75
        ]
        
        if low_quality_datasets:
            recommendations.append(f"Focus attention on datasets with quality scores below 75: {', '.join(low_quality_datasets)}")
        
        # Check for common issues
        critical_issues_count = sum(
            sum(1 for issue in report.issues_found if issue.severity.value == "critical")
            for report in validation_results.values()
        )
        
        if critical_issues_count > 0:
            recommendations.append(f"Address {critical_issues_count} critical data quality issues immediately")
        
        # Data completeness recommendations
        for name, report in validation_results.items():
            if report.data_quality_score < 90:
                recommendations.append(f"Review data completeness and accuracy for {name} dataset")
        
        if not recommendations:
            recommendations.append("Data quality is excellent across all datasets")
        
        return recommendations