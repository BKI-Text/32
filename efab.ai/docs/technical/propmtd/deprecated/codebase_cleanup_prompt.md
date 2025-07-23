# Claude.code Codebase Cleanup Analysis Prompt

**For use with Claude.code in Cursor IDE with MCP tools**

Please analyze and clean up this codebase using available MCP tools and Cursor IDE capabilities. Execute the following systematic cleanup process:

## MCP Tool Utilization Strategy

### File System Analysis
- Use `mcp-filesystem` to scan directory structure and identify cleanup targets
- Leverage `mcp-git` to analyze commit history and identify stale files
- Utilize `mcp-search` to find code patterns, unused imports, and dead code

### Code Quality & Standards Analysis
```
EXECUTE: Use MCP tools to:
1. Scan for dead/unused code, imports, and variables
2. Identify inconsistent formatting and indentation patterns
3. Find non-standard naming conventions across the codebase
4. Locate missing documentation and undocumented complex logic
5. Detect inconsistent code style patterns
```

### Structure & Organization Optimization
```
ANALYZE with Cursor IDE:
1. Map directory structure logic and identify reorganization opportunities
2. Extract repeated code patterns for modularization
3. Identify separation of concerns violations
4. Detect circular dependencies using import analysis
5. Find functionality that should be consolidated
```

### Performance & Efficiency Assessment
```
PROFILE using available tools:
1. Identify inefficient algorithms and database queries
2. Find redundant operations and unnecessary loops
3. Locate missing error handling and logging
4. Identify missing input validation points
```

### Security & Best Practices Audit
```
SECURITY SCAN:
1. Search for hardcoded secrets and sensitive data
2. Identify security vulnerabilities using pattern matching
3. Check dependency versions for known vulnerabilities
4. Find deprecated functions requiring updates
```

### Maintainability Improvements
```
REFACTOR ANALYSIS:
1. Identify overly complex functions for decomposition
2. Find missing type hints/annotations
3. Detect inconsistent error handling patterns
4. Locate inflexible configuration implementations
```

## Claude.code Execution Plan

### Phase 1: Discovery & Analysis
```python
# Use MCP tools to gather comprehensive codebase metrics
analysis_results = {
    "file_count": scan_filesystem(),
    "dead_code": find_unused_code(),
    "security_issues": scan_security_patterns(),
    "performance_bottlenecks": analyze_performance(),
    "style_inconsistencies": check_code_style(),
    "missing_documentation": find_undocumented_code()
}
```

### Phase 2: Automated Cleanup Execution
```
FOR EACH cleanup category:
1. BACKUP affected files using git
2. IMPLEMENT cleanup changes
3. VALIDATE changes don't break functionality
4. COMMIT changes with descriptive messages
5. PROCEED to next category
```

### Phase 3: Validation & Testing
```
VALIDATION_SEQUENCE:
1. Run all existing tests to ensure no regression
2. Perform static analysis to verify improvements
3. Check that all imports resolve correctly
4. Validate that application still starts/runs
5. Confirm no new security vulnerabilities introduced
```

## MCP Tool Commands

### File System Operations
```bash
# Use mcp-filesystem to identify cleanup targets
find_unused_files(exclude_patterns=[".git", "node_modules", "__pycache__"])
analyze_file_sizes(threshold="1MB")
scan_duplicate_files(hash_algorithm="md5")
```

### Git History Analysis
```bash
# Use mcp-git to understand file evolution
git_file_history(show_unused=True, older_than="6months")
git_blame_analysis(find_dead_code=True)
git_branch_analysis(find_orphaned_files=True)
```

### Code Pattern Search
```bash
# Use mcp-search for pattern-based cleanup
search_patterns([
    "TODO:", "FIXME:", "HACK:", "XXX:",
    "console.log", "print(", "debugger;",
    "import.*unused", "from.*import.*unused"
])
```

## Cursor IDE Integration

### Automated Refactoring
- Utilize Cursor's AI-powered refactoring for function decomposition
- Use intelligent code completion for standardizing naming conventions
- Leverage multi-cursor editing for bulk pattern replacements

### Code Quality Checks
- Enable real-time linting and formatting
- Use integrated type checking for missing annotations
- Implement automated import organization

## Output Specification

Provide detailed cleanup report with:
```markdown
# Codebase Cleanup Execution Report

## Summary
- Files analyzed: [count]
- Issues identified: [count]
- Automated fixes applied: [count]
- Manual review required: [count]

## Changes Applied
### Dead Code Removal
- [List of files/functions removed]
- [Rationale for each removal]

### Style Standardization
- [Formatting changes applied]
- [Naming convention updates]

### Performance Optimizations
- [Algorithmic improvements]
- [Resource usage optimizations]

### Security Enhancements
- [Vulnerabilities fixed]
- [Best practices implemented]

## Validation Results
- All tests: [PASS/FAIL]
- Static analysis: [PASS/FAIL]
- Security scan: [PASS/FAIL]

## Recommendations for Manual Review
[Items requiring human judgment]
```

Execute this cleanup systematically using the full capabilities of Claude.code, Cursor IDE, and available MCP tools to achieve professional-grade code quality.