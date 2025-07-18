# Claude.code Cleanup & Optimization Engine

**For use with Claude.code in Cursor IDE with MCP tools**

Execute intelligent codebase cleanup and performance optimization using Claude.code's analysis capabilities and coordinated MCP tool execution. This prompt supports three optimization levels: **Maintenance**, **Performance**, and **Transformation**.

## 🎯 Optimization Level Selection

### Maintenance Level
- **Duration**: 30-90 minutes
- **Scope**: Code cleanup, style standardization, basic optimizations
- **Use Case**: Regular maintenance, code quality improvement, technical debt reduction
- **Risk**: Low - Safe automated cleanups

### Performance Level (Default)
- **Duration**: 2-6 hours  
- **Scope**: Performance optimization, architectural improvements, comprehensive cleanup
- **Use Case**: Performance issues, significant refactoring, system optimization
- **Risk**: Medium - Requires validation and testing

### Transformation Level
- **Duration**: 6-24 hours
- **Scope**: Major architectural changes, technology upgrades, comprehensive modernization
- **Use Case**: Legacy modernization, architectural migration, complete system overhaul
- **Risk**: High - Requires extensive testing and validation

## 📋 Prerequisites

### Required Tools
- **Claude.code** in Cursor IDE
- **MCP Tools**: Minimum `mcp-filesystem`, `mcp-git`, `mcp-search`
- **Version Control**: Git with clean working directory
- **Backup**: Automated backup before major changes

### Optional Tools (Enhanced Features)
- `mcp-profiler` - Performance analysis and optimization
- `mcp-analyzer` - Code complexity and quality analysis
- `mcp-database` - Database query optimization
- `mcp-security` - Security vulnerability detection

## 🔧 Cleanup & Optimization Framework

### Phase 1: Comprehensive Analysis
```python
def analyze_optimization_opportunities(optimization_level="performance"):
    """Analyze codebase for cleanup and optimization opportunities"""
    
    # Core analysis (all levels)
    analysis_results = {
        "dead_code_detection": mcp_search.find_unused_code(),
        "code_style_issues": mcp_analyzer.style_inconsistencies(),
        "duplicate_code": mcp_analyzer.find_duplicates(),
        "simple_optimizations": mcp_analyzer.basic_optimizations(),
        "import_cleanup": mcp_analyzer.unused_imports()
    }
    
    # Enhanced analysis (Performance/Transformation)
    if optimization_level in ["performance", "transformation"]:
        enhanced_analysis = {
            "performance_bottlenecks": mcp_profiler.identify_bottlenecks(),
            "memory_inefficiencies": mcp_profiler.memory_analysis(),
            "database_optimization": mcp_database.query_optimization(),
            "algorithm_improvements": mcp_analyzer.algorithmic_complexity(),
            "architectural_issues": mcp_analyzer.architectural_problems()
        }
        analysis_results.update(enhanced_analysis)
    
    # Transformation analysis
    if optimization_level == "transformation":
        transformation_analysis = {
            "legacy_code_patterns": mcp_analyzer.legacy_pattern_detection(),
            "modernization_opportunities": mcp_analyzer.modernization_candidates(),
            "technology_upgrades": mcp_analyzer.technology_migration_opportunities(),
            "architectural_migrations": mcp_analyzer.architectural_improvements(),
            "security_modernization": mcp_security.security_upgrade_opportunities()
        }
        analysis_results.update(transformation_analysis)
    
    return analysis_results
```

### Phase 2: Intelligent Optimization Planning
```python
def create_optimization_plan(analysis_results, optimization_level):
    """Generate comprehensive optimization strategy using Claude.code"""
    
    # Use Claude.code semantic understanding
    optimization_strategy = cursor_ai.create_optimization_plan([
        analysis_results,
        optimization_level,
        risk_assessment=assess_optimization_risks(),
        impact_analysis=analyze_change_impact(),
        dependency_mapping=analyze_change_dependencies()
    ])
    
    # Prioritize optimization tasks by risk and impact
    prioritized_optimizations = cursor_ai.prioritize_optimizations([
        optimization_strategy,
        risk_tolerance=get_risk_tolerance(optimization_level),
        performance_impact=calculate_performance_impact(),
        maintenance_benefit=calculate_maintenance_benefit()
    ])
    
    return prioritized_optimizations
```

### Phase 3: Automated Optimization Execution
```python
def execute_optimization_plan(prioritized_optimizations, optimization_level):
    """Execute optimizations with safety checks and validation"""
    
    optimization_results = []
    
    for optimization_phase in prioritized_optimizations:
        phase_results = []
        
        # Create checkpoint before phase
        checkpoint = create_optimization_checkpoint()
        
        for optimization in optimization_phase:
            try:
                # Pre-optimization validation
                pre_state = capture_system_state()
                
                # Execute optimization using Claude.code
                optimization_result = cursor_ai.apply_optimization(
                    optimization=optimization,
                    context=gather_optimization_context(optimization),
                    safety_level=get_safety_level(optimization_level),
                    automated_testing=True
                )
                
                # Validate optimization results
                validation_result = validate_optimization(
                    pre_state, 
                    optimization_result, 
                    optimization_level
                )
                
                if validation_result.success:
                    commit_optimization(optimization, optimization_result)
                    phase_results.append(optimization_result)
                else:
                    rollback_optimization(optimization, pre_state)
                    handle_optimization_failure(optimization, validation_result)
                    
            except Exception as e:
                rollback_to_checkpoint(checkpoint)
                handle_optimization_exception(optimization, e)
                break
        
        optimization_results.append(phase_results)
    
    return optimization_results
```

## 🧹 Cleanup Categories

### 1. Code Quality Cleanup
```python
def execute_code_quality_cleanup(optimization_level):
    """Comprehensive code quality improvements"""
    
    # Base cleanup (all levels)
    quality_cleanup = {
        "remove_dead_code": remove_unused_functions_and_variables(),
        "standardize_formatting": apply_consistent_code_formatting(),
        "fix_naming_conventions": standardize_naming_across_codebase(),
        "remove_unused_imports": clean_up_import_statements(),
        "fix_code_smells": address_basic_code_quality_issues()
    }
    
    # Enhanced cleanup (Performance/Transformation)
    if optimization_level in ["performance", "transformation"]:
        enhanced_cleanup = {
            "extract_duplicate_code": consolidate_duplicate_implementations(),
            "decompose_large_functions": break_down_complex_functions(),
            "improve_error_handling": standardize_error_handling_patterns(),
            "add_missing_documentation": generate_missing_docstrings(),
            "optimize_imports": reorganize_and_optimize_imports()
        }
        quality_cleanup.update(enhanced_cleanup)
    
    # Transformation cleanup
    if optimization_level == "transformation":
        transformation_cleanup = {
            "modernize_code_patterns": update_legacy_patterns(),
            "upgrade_deprecated_apis": replace_deprecated_functions(),
            "implement_best_practices": apply_modern_coding_standards(),
            "refactor_architecture": improve_architectural_structure(),
            "add_type_annotations": add_comprehensive_type_hints()
        }
        quality_cleanup.update(transformation_cleanup)
    
    return quality_cleanup
```

### 2. Performance Optimization
```python
def execute_performance_optimization(optimization_level):
    """Comprehensive performance optimization"""
    
    # Base optimizations (all levels)
    performance_optimizations = {
        "optimize_loops": improve_loop_efficiency(),
        "optimize_data_structures": use_appropriate_data_structures(),
        "cache_expensive_operations": implement_result_caching(),
        "optimize_string_operations": improve_string_manipulation(),
        "reduce_object_creation": minimize_unnecessary_object_creation()
    }
    
    # Enhanced optimizations (Performance/Transformation)
    if optimization_level in ["performance", "transformation"]:
        enhanced_optimizations = {
            "database_query_optimization": optimize_database_interactions(),
            "algorithm_complexity_reduction": improve_algorithmic_efficiency(),
            "memory_usage_optimization": reduce_memory_footprint(),
            "io_operation_optimization": optimize_file_and_network_operations(),
            "concurrent_processing": implement_parallel_processing()
        }
        performance_optimizations.update(enhanced_optimizations)
    
    # Transformation optimizations
    if optimization_level == "transformation":
        transformation_optimizations = {
            "architectural_performance": optimize_system_architecture(),
            "distributed_computing": implement_distributed_processing(),
            "advanced_caching": implement_multi_level_caching(),
            "load_balancing": implement_load_distribution(),
            "resource_pooling": implement_resource_pooling()
        }
        performance_optimizations.update(transformation_optimizations)
    
    return performance_optimizations
```

### 3. Security Optimization
```python
def execute_security_optimization(optimization_level):
    """Comprehensive security improvements"""
    
    # Base security (all levels)
    security_optimizations = {
        "fix_basic_vulnerabilities": address_common_security_issues(),
        "improve_input_validation": enhance_input_sanitization(),
        "secure_configuration": secure_configuration_management(),
        "update_dependencies": update_vulnerable_dependencies(),
        "implement_security_headers": add_security_headers()
    }
    
    # Enhanced security (Performance/Transformation)
    if optimization_level in ["performance", "transformation"]:
        enhanced_security = {
            "comprehensive_vulnerability_scan": fix_all_security_vulnerabilities(),
            "implement_security_patterns": apply_security_design_patterns(),
            "enhance_authentication": improve_authentication_mechanisms(),
            "implement_authorization": enhance_authorization_controls(),
            "audit_logging": implement_comprehensive_audit_logging()
        }
        security_optimizations.update(enhanced_security)
    
    # Transformation security
    if optimization_level == "transformation":
        transformation_security = {
            "zero_trust_architecture": implement_zero_trust_security(),
            "advanced_threat_detection": implement_threat_monitoring(),
            "security_automation": implement_automated_security_testing(),
            "compliance_implementation": implement_compliance_controls(),
            "incident_response": implement_incident_response_procedures()
        }
        security_optimizations.update(transformation_security)
    
    return security_optimizations
```

### 4. Infrastructure Optimization
```python
def execute_infrastructure_optimization(optimization_level):
    """Infrastructure and deployment optimizations"""
    
    # Base infrastructure (all levels)
    infrastructure_optimizations = {
        "optimize_build_process": improve_build_performance(),
        "optimize_dependencies": reduce_dependency_bloat(),
        "improve_deployment": streamline_deployment_process(),
        "optimize_configuration": improve_configuration_management(),
        "enhance_monitoring": implement_basic_monitoring()
    }
    
    # Enhanced infrastructure (Performance/Transformation)
    if optimization_level in ["performance", "transformation"]:
        enhanced_infrastructure = {
            "implement_ci_cd": implement_continuous_integration(),
            "optimize_resource_usage": optimize_resource_consumption(),
            "implement_scaling": implement_auto_scaling(),
            "optimize_networking": optimize_network_performance(),
            "implement_backup": implement_automated_backup()
        }
        infrastructure_optimizations.update(enhanced_infrastructure)
    
    # Transformation infrastructure
    if optimization_level == "transformation":
        transformation_infrastructure = {
            "cloud_native_migration": migrate_to_cloud_native(),
            "microservices_architecture": implement_microservices(),
            "container_orchestration": implement_container_orchestration(),
            "service_mesh": implement_service_mesh(),
            "observability_platform": implement_comprehensive_observability()
        }
        infrastructure_optimizations.update(transformation_infrastructure)
    
    return infrastructure_optimizations
```

## 🎨 Safety & Validation Framework

### Automated Safety Checks
```python
def implement_safety_framework(optimization_level):
    """Implement comprehensive safety checks"""
    
    safety_checks = {
        "backup_creation": create_automated_backups(),
        "rollback_capability": implement_rollback_procedures(),
        "incremental_changes": implement_incremental_optimization(),
        "validation_testing": implement_automated_validation(),
        "impact_assessment": assess_change_impact()
    }
    
    # Enhanced safety (Performance/Transformation)
    if optimization_level in ["performance", "transformation"]:
        enhanced_safety = {
            "comprehensive_testing": implement_comprehensive_test_suite(),
            "performance_regression_detection": implement_performance_monitoring(),
            "security_validation": implement_security_testing(),
            "integration_testing": implement_integration_validation(),
            "user_acceptance_testing": implement_user_testing()
        }
        safety_checks.update(enhanced_safety)
    
    return safety_checks
```

### Validation Metrics
```python
def define_validation_metrics(optimization_level):
    """Define success metrics for optimization"""
    
    # Base metrics (all levels)
    validation_metrics = {
        "code_quality_improvement": measure_code_quality_metrics(),
        "performance_improvement": measure_performance_gains(),
        "security_enhancement": measure_security_improvements(),
        "maintainability_improvement": measure_maintainability_gains(),
        "test_coverage_improvement": measure_test_coverage_gains()
    }
    
    # Enhanced metrics (Performance/Transformation)
    if optimization_level in ["performance", "transformation"]:
        enhanced_metrics = {
            "scalability_improvement": measure_scalability_gains(),
            "reliability_improvement": measure_reliability_gains(),
            "operational_efficiency": measure_operational_improvements(),
            "resource_optimization": measure_resource_savings(),
            "user_experience_improvement": measure_ux_improvements()
        }
        validation_metrics.update(enhanced_metrics)
    
    return validation_metrics
```

## 🚀 Usage Examples

### Maintenance Level Cleanup
```python
# Regular maintenance cleanup
cleanup_result = execute_cleanup_optimization(
    level="maintenance",
    focus_areas=["code_quality", "style_standardization", "basic_optimizations"],
    safety_level="high",
    automated_testing=True
)
```

### Performance Level Optimization
```python
# Performance optimization
optimization_result = execute_cleanup_optimization(
    level="performance",
    focus_areas=["performance", "architecture", "database_optimization"],
    target_improvements={"response_time": "50%", "memory_usage": "30%"},
    comprehensive_testing=True
)
```

### Transformation Level Modernization
```python
# Complete system transformation
transformation_result = execute_cleanup_optimization(
    level="transformation",
    focus_areas=["modernization", "architecture", "security", "performance"],
    migration_targets=["cloud_native", "microservices", "modern_frameworks"],
    comprehensive_validation=True
)
```

## 📊 Output Specifications

### Optimization Report
```markdown
# Cleanup & Optimization Report

## Executive Summary
- **Optimization Level**: [Selected level]
- **Duration**: [Actual time taken]
- **Optimizations Applied**: [Count/Total]
- **Performance Improvement**: [Percentage]
- **Quality Improvement**: [Metrics]

## Code Quality Improvements
### Dead Code Removal
- **Files cleaned**: [Count]
- **Functions removed**: [Count]
- **Lines of code reduced**: [Count]

### Style Standardization
- **Files formatted**: [Count]
- **Naming conventions fixed**: [Count]
- **Import statements cleaned**: [Count]

### Duplicate Code Elimination
- **Duplicates found**: [Count]
- **Code consolidated**: [Percentage]
- **Maintainability improvement**: [Score]

## Performance Optimizations
### Algorithm Improvements
- **Functions optimized**: [Count]
- **Complexity reductions**: [Details]
- **Performance gains**: [Percentage]

### Database Optimization
- **Queries optimized**: [Count]
- **Query performance improvement**: [Percentage]
- **Database load reduction**: [Percentage]

### Memory Optimization
- **Memory usage reduction**: [Percentage]
- **Object creation optimization**: [Count]
- **Memory leak fixes**: [Count]

## Security Enhancements
- **Vulnerabilities fixed**: [Count]
- **Security patterns implemented**: [Count]
- **Configuration secured**: [Count]
- **Dependencies updated**: [Count]

## Infrastructure Improvements
- **Build time improvement**: [Percentage]
- **Deployment optimization**: [Details]
- **Resource usage optimization**: [Percentage]
- **Monitoring enhancements**: [Count]

## Validation Results
- **All tests passed**: [Status]
- **Performance benchmarks**: [Results]
- **Security scan**: [Results]
- **Code quality metrics**: [Scores]
```

## 🔄 Continuous Optimization

### Ongoing Optimization Monitoring
```python
# Establish continuous optimization monitoring
optimization_monitoring = {
    "code_quality_degradation_detection": True,
    "performance_regression_monitoring": True,
    "security_vulnerability_scanning": True,
    "technical_debt_tracking": True,
    "optimization_opportunity_detection": True
}
```

### Automated Optimization Scheduling
```python
# Schedule regular optimization cycles
optimization_schedule = {
    "maintenance_cleanup": "weekly",
    "performance_optimization": "monthly",
    "security_updates": "continuous",
    "dependency_updates": "weekly",
    "architectural_reviews": "quarterly"
}
```

## 🛠️ Error Handling & Recovery

### Optimization Failure Recovery
```python
def handle_optimization_failure(optimization, validation_result):
    """Handle optimization failures gracefully"""
    
    recovery_strategies = {
        "rollback_changes": rollback_to_previous_state(),
        "partial_optimization": apply_safe_subset_of_changes(),
        "alternative_approach": try_alternative_optimization(),
        "manual_review": flag_for_manual_intervention()
    }
    
    return execute_recovery_strategy(optimization, validation_result, recovery_strategies)
```

### Validation Failure Handling
```python
def handle_validation_failure(validation_type, failure_details):
    """Handle validation failures"""
    
    remediation_actions = {
        "test_failure": fix_broken_tests(),
        "performance_regression": revert_performance_changes(),
        "security_issue": address_security_problems(),
        "integration_failure": fix_integration_issues()
    }
    
    return execute_remediation(validation_type, failure_details, remediation_actions)
```

## 📈 Success Metrics

### Optimization Effectiveness
- **Code Quality Score**: Before/after comparison
- **Performance Metrics**: Response time, memory usage, throughput
- **Security Posture**: Vulnerability count, compliance score
- **Maintainability Index**: Code complexity, documentation coverage
- **Technical Debt**: Debt ratio, remediation progress

### Long-term Benefits
- **Development Velocity**: Feature delivery speed
- **Bug Reduction**: Bug occurrence rate
- **Operational Efficiency**: Deployment success rate, downtime reduction
- **Cost Savings**: Resource utilization, maintenance costs
- **Developer Satisfaction**: Code quality perception, development experience

Execute this comprehensive cleanup and optimization using the appropriate level for your needs, leveraging the full intelligence of Claude.code and available MCP tools to achieve significant improvements in code quality, performance, and maintainability.