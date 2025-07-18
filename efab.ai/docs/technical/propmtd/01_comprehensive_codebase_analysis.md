# Claude.code Comprehensive Codebase Analysis Engine

**For use with Claude.code in Cursor IDE with MCP tools**

Execute intelligent, multi-level codebase analysis using Claude.code semantic understanding and coordinated MCP tool execution. This prompt supports three analysis depths: **Standard**, **Deep**, and **Forensic**.

## 🎯 Analysis Depth Selection

### Standard Analysis (Default)
- **Duration**: 5-15 minutes
- **Scope**: Core architecture, code quality, basic security
- **Use Case**: Regular health checks, onboarding new developers
- **MCP Tools**: `mcp-filesystem`, `mcp-git`, `mcp-search`

### Deep Analysis
- **Duration**: 30-60 minutes  
- **Scope**: Performance profiling, dependency analysis, pattern recognition
- **Use Case**: Performance optimization, architectural planning
- **MCP Tools**: All standard tools + `mcp-profiler`, `mcp-dependencies`, `mcp-analyzer`

### Forensic Analysis
- **Duration**: 1-3 hours
- **Scope**: Complete system forensics, security audit, migration planning
- **Use Case**: Security audits, legacy system analysis, critical architecture decisions
- **MCP Tools**: Complete MCP tool suite with maximum depth

## 📋 Prerequisites

### Required Tools
- **Claude.code** in Cursor IDE
- **MCP Tools**: Minimum `mcp-filesystem`, `mcp-git`, `mcp-search`
- **Git Repository**: Working directory with git history

### Optional Tools (Enhanced Features)
- `mcp-profiler` - Performance analysis
- `mcp-security` - Security vulnerability scanning
- `mcp-dependencies` - Dependency relationship mapping
- `mcp-analyzer` - Code complexity analysis
- `mcp-database` - Database optimization analysis

## 🔧 Analysis Execution Framework

### Phase 1: Discovery & Mapping
```python
def execute_codebase_discovery(depth_level="standard"):
    """Execute codebase discovery based on selected depth"""
    
    # Core discovery (all levels)
    core_discovery = {
        "filesystem_structure": mcp_filesystem.analyze_structure(),
        "git_history_analysis": mcp_git.evolution_patterns(),
        "code_organization": mcp_search.pattern_analysis(),
        "entry_points": identify_application_entry_points(),
        "configuration_files": discover_configuration_management()
    }
    
    # Enhanced discovery (Deep/Forensic)
    if depth_level in ["deep", "forensic"]:
        enhanced_discovery = {
            "dependency_ecosystem": mcp_dependencies.full_mapping(),
            "performance_baseline": mcp_profiler.baseline_analysis(),
            "security_landscape": mcp_security.comprehensive_scan(),
            "code_complexity": mcp_analyzer.complexity_metrics()
        }
        core_discovery.update(enhanced_discovery)
    
    # Forensic-level discovery
    if depth_level == "forensic":
        forensic_discovery = {
            "data_flow_mapping": mcp_data_analyzer.complete_flows(),
            "execution_tracing": mcp_tracer.exhaustive_paths(),
            "memory_analysis": mcp_memory.usage_patterns(),
            "infrastructure_mapping": analyze_infrastructure_dependencies()
        }
        core_discovery.update(forensic_discovery)
    
    return core_discovery
```

### Phase 2: Intelligence Synthesis
```python
def synthesize_codebase_intelligence(discovery_results, depth_level):
    """Generate intelligent insights from discovery data"""
    
    # Use Claude.code semantic understanding
    intelligence_synthesis = cursor_ai.analyze_codebase([
        "architectural_patterns",     # Identify design patterns
        "code_quality_assessment",   # Overall code quality
        "maintainability_scoring",   # Maintenance complexity
        "technical_debt_mapping",    # Areas needing improvement
        "security_posture",          # Security implementation status
        "performance_characteristics" # Performance profile
    ])
    
    # Generate contextual insights
    contextual_insights = {
        "system_overview": generate_executive_summary(),
        "architectural_assessment": analyze_architecture_quality(),
        "development_workflow": understand_development_patterns(),
        "operational_characteristics": assess_operational_readiness(),
        "improvement_opportunities": identify_optimization_areas()
    }
    
    return {
        "semantic_analysis": intelligence_synthesis,
        "contextual_insights": contextual_insights,
        "discovery_data": discovery_results
    }
```

### Phase 3: Knowledge Artifact Generation
```python
def generate_analysis_artifacts(analysis_results, depth_level):
    """Create comprehensive documentation and visualizations"""
    
    # Standard artifacts (all levels)
    artifacts = {
        "system_overview": create_executive_summary(),
        "architecture_diagram": generate_architecture_visualization(),
        "developer_guide": create_onboarding_documentation(),
        "api_documentation": extract_api_specifications()
    }
    
    # Enhanced artifacts (Deep/Forensic)
    if depth_level in ["deep", "forensic"]:
        enhanced_artifacts = {
            "performance_analysis": create_performance_report(),
            "security_assessment": generate_security_report(),
            "dependency_map": create_dependency_visualization(),
            "refactoring_recommendations": generate_improvement_plan()
        }
        artifacts.update(enhanced_artifacts)
    
    # Forensic-level artifacts
    if depth_level == "forensic":
        forensic_artifacts = {
            "technical_risk_assessment": create_risk_analysis(),
            "migration_roadmap": generate_migration_plan(),
            "compliance_audit": assess_compliance_status(),
            "operational_runbook": create_operational_guide()
        }
        artifacts.update(forensic_artifacts)
    
    return artifacts
```

## 🎨 MCP Tool Orchestration Patterns

### Sequential Analysis Pattern
```python
# For systematic analysis
discovery = mcp_filesystem.scan()
history = mcp_git.analyze(discovery)
insights = cursor_ai.synthesize(history)
artifacts = generate_documentation(insights)
```

### Parallel Analysis Pattern
```python
# For performance optimization
analysis_results = asyncio.gather([
    mcp_filesystem.analyze(),
    mcp_git.history_analysis(),
    mcp_security.scan(),
    mcp_profiler.baseline()
])
```

### Conditional Analysis Pattern
```python
# Adapt based on project type
if project_type == "web_application":
    additional_analysis = [
        mcp_api_scanner.endpoint_discovery(),
        mcp_security.web_vulnerability_scan()
    ]
elif project_type == "data_pipeline":
    additional_analysis = [
        mcp_data_analyzer.pipeline_analysis(),
        mcp_database.optimization_scan()
    ]
```

## 📊 Output Specifications

### Executive Summary (All Levels)
```markdown
# Codebase Analysis Summary

## System Overview
- **Project Type**: [Auto-detected type]
- **Primary Language**: [Language with percentage]
- **Architecture Pattern**: [Detected pattern]
- **Lines of Code**: [Count with breakdown]
- **Complexity Score**: [Calculated metric]

## Key Findings
- **Strengths**: [Top 3 positive aspects]
- **Concerns**: [Top 3 areas needing attention]  
- **Recommendations**: [Priority improvements]

## Technical Metrics
- **Code Quality**: [Score/Grade]
- **Test Coverage**: [Percentage]
- **Security Posture**: [Assessment]
- **Performance**: [Baseline metrics]
```

### Technical Deep Dive (Deep/Forensic)
```markdown
## Architecture Analysis
### Component Structure
[Auto-generated component diagram]

### Communication Patterns
[Service interaction visualization]

### Data Flow Architecture
[Data movement and transformation diagram]

## Performance Characteristics
- **Bottlenecks**: [Identified performance issues]
- **Resource Usage**: [Memory, CPU, I/O patterns]
- **Optimization Opportunities**: [Specific recommendations]

## Security Assessment
- **Vulnerabilities**: [Identified security issues]
- **Compliance**: [Standards adherence]
- **Recommendations**: [Security improvements]
```

### Forensic Report (Forensic Only)
```markdown
## Risk Assessment
### Technical Risks
- [Detailed risk analysis with impact assessment]

### Operational Risks
- [Deployment and maintenance risks]

### Business Risks
- [Impact on business operations]

## Migration Considerations
- **Legacy Components**: [Components requiring modernization]
- **Breaking Changes**: [Potential compatibility issues]
- **Timeline Estimates**: [Migration effort estimates]
```

## 🚀 Usage Examples

### Standard Analysis
```python
# Quick health check
analysis_result = execute_codebase_analysis(
    depth="standard",
    focus_areas=["architecture", "code_quality"],
    output_format="summary"
)
```

### Deep Analysis for Performance
```python
# Performance optimization planning
analysis_result = execute_codebase_analysis(
    depth="deep",
    focus_areas=["performance", "architecture", "dependencies"],
    include_recommendations=True,
    benchmark_against="industry_standards"
)
```

### Forensic Analysis for Security Audit
```python
# Comprehensive security assessment
analysis_result = execute_codebase_analysis(
    depth="forensic",
    focus_areas=["security", "compliance", "risk_assessment"],
    compliance_standards=["OWASP", "GDPR", "SOC2"],
    generate_remediation_plan=True
)
```

## 🔄 Continuous Monitoring Setup

### Ongoing Analysis Automation
```python
# Establish continuous analysis
monitoring_config = {
    "analysis_frequency": "weekly",
    "depth_level": "standard",
    "alert_on_degradation": True,
    "automated_reporting": True,
    "trend_analysis": True
}
```

### Quality Gate Integration
```python
# Integrate with CI/CD
quality_gates = {
    "code_quality_threshold": 8.0,
    "security_vulnerability_tolerance": "low",
    "performance_regression_detection": True,
    "architecture_compliance_check": True
}
```

## 🛠️ Troubleshooting Guide

### Common Issues
- **MCP Tool Not Found**: Graceful degradation to available tools
- **Large Codebase Timeout**: Automatic chunking and batching
- **Memory Constraints**: Streaming analysis for large files
- **Incomplete Analysis**: Partial results with clear status

### Error Handling
```python
try:
    analysis_result = execute_analysis()
except MCPToolUnavailable as e:
    fallback_analysis = execute_fallback_analysis()
except TimeoutError:
    chunked_analysis = execute_chunked_analysis()
```

## 📈 Success Metrics

### Analysis Quality Indicators
- **Coverage**: Percentage of codebase analyzed
- **Accuracy**: Validation against known issues
- **Completeness**: All requested areas covered
- **Actionability**: Clear, implementable recommendations

### Performance Metrics  
- **Analysis Speed**: Time per thousand lines of code
- **Resource Usage**: Memory and CPU consumption
- **Artifact Quality**: Documentation completeness and accuracy

Execute this comprehensive analysis using the appropriate depth level for your needs, leveraging the full intelligence of Claude.code and available MCP tools.