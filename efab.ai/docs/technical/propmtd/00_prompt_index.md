# Claude.code Development Prompts Collection - Optimized Edition

**For use with Claude.code in Cursor IDE with MCP (Model Context Protocol) tools**

This streamlined collection contains 3 comprehensive prompts designed for Claude.code AI development automation, replacing the previous 8 prompts with better organization and reduced redundancy.

## 🎯 Optimized Prompt Collection

### 1. **Comprehensive Codebase Analysis Engine**
**File**: `01_comprehensive_codebase_analysis.md`
- **Purpose**: Multi-level intelligent codebase analysis and understanding
- **Modes**: Standard (15 min) | Deep (60 min) | Forensic (3 hours)
- **Use Cases**: 
  - System documentation and onboarding
  - Architecture assessment and planning
  - Security audits and compliance validation
  - Legacy system analysis and modernization planning
- **MCP Tools**: `mcp-filesystem`, `mcp-git`, `mcp-search`, `mcp-analyzer`, `mcp-security`
- **Key Features**:
  - Adaptive analysis depth based on requirements
  - Automated documentation generation
  - Visual architecture mapping
  - Security posture assessment
  - Performance baseline establishment

### 2. **AI Development Completion Engine**
**File**: `02_ai_development_completion.md`
- **Purpose**: Comprehensive AI-driven development completion to production standards
- **Modes**: Sprint (4 hours) | Production (12 hours) | Enterprise (48 hours)
- **Use Cases**:
  - MVP and prototype completion
  - Production deployment preparation
  - Enterprise-grade implementation
  - Commercial software development
- **MCP Tools**: `mcp-analyzer`, `mcp-security`, `mcp-profiler`, `mcp-testing`
- **Key Features**:
  - Automated gap analysis and implementation
  - Quality gates and commercial standards compliance
  - Security implementation automation
  - Performance optimization integration
  - Comprehensive testing generation

### 3. **Cleanup & Optimization Engine**
**File**: `03_cleanup_optimization_engine.md`
- **Purpose**: Intelligent code cleanup and performance optimization
- **Modes**: Maintenance (90 min) | Performance (6 hours) | Transformation (24 hours)
- **Use Cases**:
  - Regular code maintenance and quality improvement
  - Performance bottleneck resolution
  - Legacy system modernization
  - Architectural transformation
- **MCP Tools**: `mcp-filesystem`, `mcp-profiler`, `mcp-analyzer`, `mcp-database`
- **Key Features**:
  - Automated dead code removal
  - Performance optimization with benchmarking
  - Security vulnerability remediation
  - Architecture modernization
  - Comprehensive safety checks and rollback

## 📋 Quick Selection Guide

### 🔍 **Need to Understand a Codebase?**
**Use**: Comprehensive Codebase Analysis Engine
```
- New project onboarding → Standard mode
- Performance issues investigation → Deep mode  
- Security audit or compliance → Forensic mode
- Legacy system assessment → Forensic mode
```

### 🚀 **Need to Complete Development?**
**Use**: AI Development Completion Engine
```
- MVP or prototype → Sprint mode
- Production deployment → Production mode
- Enterprise or regulated industry → Enterprise mode
- Commercial software release → Production mode
```

### 🧹 **Need to Improve Code Quality?**
**Use**: Cleanup & Optimization Engine
```
- Regular maintenance → Maintenance mode
- Performance problems → Performance mode
- Legacy modernization → Transformation mode
- Technical debt reduction → Performance mode
```

## 🛠️ MCP Tool Requirements

### Essential Tools (All Prompts)
- **mcp-filesystem**: File system operations and analysis
- **mcp-git**: Version control analysis and operations
- **mcp-search**: Pattern search and code analysis

### Enhanced Tools (Optional but Recommended)
- **mcp-analyzer**: Code complexity and quality analysis
- **mcp-profiler**: Performance analysis and optimization
- **mcp-security**: Security vulnerability scanning
- **mcp-testing**: Test coverage and quality analysis
- **mcp-database**: Database optimization analysis
- **mcp-dependencies**: Dependency management analysis

### Tool Availability Check
```python
# Check available MCP tools before starting
available_tools = check_mcp_tools([
    "mcp-filesystem",    # Required
    "mcp-git",          # Required
    "mcp-search",       # Required
    "mcp-analyzer",     # Recommended
    "mcp-profiler",     # Recommended
    "mcp-security",     # Recommended
])

# Adapt prompt execution based on available tools
if "mcp-profiler" not in available_tools:
    print("Performance analysis will be limited")
if "mcp-security" not in available_tools:
    print("Security scanning will be basic")
```

## 🎨 Usage Patterns

### Sequential Workflow (Recommended)
```python
# Comprehensive development workflow
workflow = [
    "01_comprehensive_codebase_analysis.md",  # Understand current state
    "02_ai_development_completion.md",        # Complete development
    "03_cleanup_optimization_engine.md"      # Optimize and clean
]
```

### Parallel Analysis (Advanced)
```python
# Concurrent analysis for large projects
parallel_analysis = asyncio.gather([
    execute_codebase_analysis(depth="deep"),
    execute_performance_analysis(),
    execute_security_analysis()
])
```

### Incremental Improvement (Maintenance)
```python
# Regular maintenance cycle
maintenance_cycle = {
    "weekly": "03_cleanup_optimization_engine.md (maintenance)",
    "monthly": "01_comprehensive_codebase_analysis.md (standard)",
    "quarterly": "02_ai_development_completion.md (production)"
}
```

## ⚙️ Configuration Guidelines

### Environment Setup
```python
# Recommended environment configuration
environment_config = {
    "cursor_ide": "latest_version",
    "mcp_tools": "complete_suite",
    "git_repository": "clean_working_directory",
    "backup_strategy": "automated_pre_execution",
    "testing_framework": "project_appropriate"
}
```

### Safety Settings
```python
# Safety configuration by risk level
safety_config = {
    "low_risk": {
        "backup_required": True,
        "testing_level": "smoke_tests",
        "validation_depth": "basic"
    },
    "medium_risk": {
        "backup_required": True,
        "testing_level": "comprehensive",
        "validation_depth": "thorough"
    },
    "high_risk": {
        "backup_required": True,
        "testing_level": "full_suite",
        "validation_depth": "exhaustive",
        "rollback_capability": True
    }
}
```

## 🎭 Mode Selection Decision Tree

### Analysis Depth Selection
```
Need quick overview? → Standard mode (15 min)
Need detailed analysis? → Deep mode (60 min)
Need comprehensive audit? → Forensic mode (3 hours)
```

### Development Completion Selection
```
Building MVP/prototype? → Sprint mode (4 hours)
Deploying to production? → Production mode (12 hours)
Enterprise deployment? → Enterprise mode (48 hours)
```

### Optimization Level Selection
```
Regular maintenance? → Maintenance mode (90 min)
Performance issues? → Performance mode (6 hours)
Major modernization? → Transformation mode (24 hours)
```

## 📊 Expected Outcomes

### Comprehensive Codebase Analysis
- **Standard**: System overview, architecture diagram, basic recommendations
- **Deep**: Performance analysis, security assessment, detailed improvement plan
- **Forensic**: Complete system audit, risk assessment, migration roadmap

### AI Development Completion  
- **Sprint**: Working MVP with basic features and testing
- **Production**: Commercial-grade implementation with full testing
- **Enterprise**: Enterprise-ready system with compliance and audit trails

### Cleanup & Optimization
- **Maintenance**: Code quality improvement, basic optimizations
- **Performance**: Significant performance improvements, architectural enhancements
- **Transformation**: Complete system modernization, technology upgrades

## 🔧 Troubleshooting Guide

### Common Issues

#### MCP Tool Unavailable
```python
# Graceful degradation
if not mcp_tool_available("mcp-profiler"):
    use_alternative_performance_analysis()
    log_limitation("Performance analysis will be basic")
```

#### Timeout Issues
```python
# Handle large codebases
if codebase_size > threshold:
    enable_chunked_processing()
    increase_timeout_limits()
```

#### Memory Constraints
```python
# Memory optimization
if memory_usage > limit:
    enable_streaming_analysis()
    reduce_concurrent_operations()
```

#### Validation Failures
```python
# Validation error handling
if validation_fails:
    rollback_changes()
    review_and_retry()
    flag_for_manual_review()
```

### Error Recovery Strategies

#### Partial Failure Recovery
```python
def handle_partial_failure(failed_component, completed_components):
    """Handle partial execution failures"""
    
    recovery_options = {
        "continue_with_limitations": use_completed_components(),
        "retry_failed_component": retry_with_different_approach(),
        "manual_intervention": flag_for_manual_completion()
    }
    
    return choose_recovery_strategy(failed_component, recovery_options)
```

#### System State Recovery
```python
def recover_system_state(checkpoint):
    """Recover from system state issues"""
    
    recovery_steps = [
        restore_from_checkpoint(checkpoint),
        validate_system_integrity(),
        resume_execution_if_safe(),
        report_recovery_status()
    ]
    
    return execute_recovery_steps(recovery_steps)
```

## 📈 Success Metrics

### Quality Indicators
- **Completeness**: Percentage of requested analysis/implementation completed
- **Accuracy**: Validation against known issues and requirements
- **Efficiency**: Time per unit of work (lines of code, features, etc.)
- **Reliability**: Success rate of automated operations

### Performance Metrics
- **Analysis Speed**: Lines of code analyzed per minute
- **Implementation Speed**: Features completed per hour
- **Optimization Impact**: Performance improvement percentage
- **Resource Utilization**: CPU, memory, and tool usage efficiency

## 🎯 Best Practices

### Pre-Execution Checklist
- [ ] Verify MCP tool availability
- [ ] Confirm git repository is clean
- [ ] Create automated backup
- [ ] Set appropriate timeout limits
- [ ] Configure safety settings

### During Execution
- [ ] Monitor progress and resource usage
- [ ] Validate intermediate results
- [ ] Handle errors gracefully
- [ ] Maintain execution logs
- [ ] Be prepared to intervene if needed

### Post-Execution
- [ ] Validate all outputs
- [ ] Run comprehensive tests
- [ ] Document changes made
- [ ] Update team on results
- [ ] Plan follow-up actions

## 📁 File Organization

```
/docs/technical/propmtd/
├── 00_prompt_index.md                    # This index file
├── 01_comprehensive_codebase_analysis.md # Multi-level analysis
├── 02_ai_development_completion.md       # Development completion
├── 03_cleanup_optimization_engine.md     # Cleanup and optimization
├── 04_mcp_tool_configuration.md          # Tool setup guide
└── examples/                             # Usage examples
    ├── analysis_examples.md
    ├── development_examples.md
    └── optimization_examples.md
```

## 🚀 Getting Started

### First-Time Setup
1. **Install Prerequisites**: Ensure Cursor IDE and MCP tools are installed
2. **Verify Tools**: Run MCP tool availability check
3. **Create Backup**: Set up automated backup strategy
4. **Choose Prompt**: Select appropriate prompt for your needs
5. **Configure Safety**: Set safety level based on risk tolerance

### Quick Start Commands
```python
# Standard codebase analysis
execute_prompt("01_comprehensive_codebase_analysis.md", mode="standard")

# MVP development completion  
execute_prompt("02_ai_development_completion.md", mode="sprint")

# Regular maintenance cleanup
execute_prompt("03_cleanup_optimization_engine.md", mode="maintenance")
```

This optimized prompt collection provides comprehensive AI-driven development automation while maintaining safety, reliability, and ease of use. Choose the appropriate prompt and mode based on your specific needs and risk tolerance.