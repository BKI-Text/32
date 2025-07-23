# MCP Tool Configuration Guide for Claude.code

**Complete setup guide for Model Context Protocol tools with Claude.code**

This guide provides comprehensive configuration instructions for MCP tools used with Claude.code in Cursor IDE, including installation, configuration, troubleshooting, and optimization.

## 🛠️ MCP Tool Overview

### What are MCP Tools?
Model Context Protocol (MCP) tools are specialized utilities that extend Claude.code's capabilities by providing:
- **Structured Data Access**: Consistent interfaces for code analysis
- **Automated Operations**: Streamlined execution of complex tasks
- **Context Awareness**: Understanding of project structure and dependencies
- **Safety Mechanisms**: Built-in validation and rollback capabilities

### Tool Categories

#### Essential Tools (Required)
- **mcp-filesystem**: File system operations and analysis
- **mcp-git**: Version control analysis and operations  
- **mcp-search**: Pattern search and code analysis

#### Enhanced Tools (Recommended)
- **mcp-analyzer**: Code complexity and quality analysis
- **mcp-profiler**: Performance analysis and optimization
- **mcp-security**: Security vulnerability scanning
- **mcp-testing**: Test coverage and quality analysis

#### Specialized Tools (Optional)
- **mcp-database**: Database optimization analysis
- **mcp-dependencies**: Dependency management analysis
- **mcp-api-scanner**: API endpoint discovery and analysis
- **mcp-docs**: Documentation generation and maintenance

## 📦 Installation Instructions

### Prerequisites
```bash
# Required software
- Node.js 18+ or Python 3.8+
- Git 2.30+
- Cursor IDE (latest version)
- Claude.code extension

# System requirements
- RAM: 4GB minimum, 8GB recommended
- Storage: 2GB free space for tools
- Network: Internet connection for installation
```

### Installation Methods

#### Method 1: Package Manager Installation (Recommended)
```bash
# Using npm (Node.js)
npm install -g @anthropic/mcp-tools

# Using pip (Python)
pip install anthropic-mcp-tools

# Using yarn
yarn global add @anthropic/mcp-tools
```

#### Method 2: Direct Download
```bash
# Download and install from GitHub
git clone https://github.com/anthropic/mcp-tools.git
cd mcp-tools
./install.sh
```

#### Method 3: Docker Installation
```bash
# Pull Docker image
docker pull anthropic/mcp-tools:latest

# Run container
docker run -v $(pwd):/workspace anthropic/mcp-tools:latest
```

### Tool-Specific Installation

#### mcp-filesystem
```bash
# Install filesystem analysis tool
npm install -g @anthropic/mcp-filesystem

# Verify installation
mcp-filesystem --version
```

#### mcp-git
```bash
# Install git analysis tool
npm install -g @anthropic/mcp-git

# Configure git access
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

#### mcp-security
```bash
# Install security scanning tool
npm install -g @anthropic/mcp-security

# Update vulnerability database
mcp-security --update-db
```

## ⚙️ Configuration

### Basic Configuration

#### Global Configuration File
Create `~/.mcp/config.json`:
```json
{
  "version": "1.0",
  "tools": {
    "mcp-filesystem": {
      "enabled": true,
      "timeout": 30000,
      "max_file_size": "10MB",
      "excluded_paths": [
        "node_modules",
        ".git",
        "*.tmp",
        "*.log"
      ]
    },
    "mcp-git": {
      "enabled": true,
      "timeout": 60000,
      "max_history": 1000,
      "analyze_branches": ["main", "develop"]
    },
    "mcp-search": {
      "enabled": true,
      "timeout": 45000,
      "max_results": 1000,
      "case_sensitive": false
    },
    "mcp-analyzer": {
      "enabled": true,
      "timeout": 120000,
      "complexity_threshold": 10,
      "quality_threshold": 7.0
    },
    "mcp-profiler": {
      "enabled": true,
      "timeout": 300000,
      "sampling_rate": 1000,
      "memory_tracking": true
    },
    "mcp-security": {
      "enabled": true,
      "timeout": 180000,
      "vulnerability_db": "latest",
      "severity_threshold": "medium"
    }
  },
  "safety": {
    "backup_enabled": true,
    "rollback_points": 5,
    "validation_level": "thorough"
  },
  "performance": {
    "concurrent_tools": 3,
    "memory_limit": "2GB",
    "cache_enabled": true
  }
}
```

#### Project-Specific Configuration
Create `.mcp/project.json` in your project root:
```json
{
  "project": {
    "name": "Beverly Knits AI Supply Chain",
    "type": "python_web_application",
    "language": "python",
    "framework": "fastapi"
  },
  "analysis": {
    "focus_areas": [
      "performance",
      "security",
      "code_quality"
    ],
    "excluded_files": [
      "test_*.py",
      "migrations/",
      "static/"
    ]
  },
  "thresholds": {
    "code_quality": 8.0,
    "test_coverage": 80,
    "security_score": 90
  }
}
```

### Environment-Specific Configuration

#### Development Environment
```json
{
  "environment": "development",
  "tools": {
    "mcp-profiler": {
      "detailed_analysis": true,
      "performance_tracking": true
    },
    "mcp-security": {
      "strict_mode": false,
      "dev_exceptions": true
    }
  },
  "safety": {
    "backup_frequency": "on_demand",
    "validation_level": "basic"
  }
}
```

#### Production Environment
```json
{
  "environment": "production",
  "tools": {
    "mcp-security": {
      "strict_mode": true,
      "comprehensive_scan": true
    },
    "mcp-profiler": {
      "optimization_focus": true,
      "resource_monitoring": true
    }
  },
  "safety": {
    "backup_frequency": "before_each_operation",
    "validation_level": "exhaustive"
  }
}
```

## 🔧 Tool-Specific Configuration

### mcp-filesystem Configuration
```json
{
  "mcp-filesystem": {
    "scan_depth": 10,
    "follow_symlinks": false,
    "ignore_patterns": [
      "*.pyc",
      "__pycache__",
      ".DS_Store",
      "Thumbs.db"
    ],
    "file_size_limits": {
      "max_file_size": "50MB",
      "warn_threshold": "10MB"
    },
    "analysis_options": {
      "calculate_checksums": true,
      "detect_duplicates": true,
      "analyze_structure": true
    }
  }
}
```

### mcp-git Configuration
```json
{
  "mcp-git": {
    "history_depth": 500,
    "include_branches": ["main", "develop", "feature/*"],
    "exclude_authors": ["automation-bot", "ci-system"],
    "analysis_options": {
      "blame_analysis": true,
      "commit_frequency": true,
      "branch_analysis": true,
      "merge_analysis": true
    },
    "performance": {
      "cache_git_log": true,
      "parallel_processing": true
    }
  }
}
```

### mcp-security Configuration
```json
{
  "mcp-security": {
    "vulnerability_sources": [
      "npm_audit",
      "safety_db",
      "snyk",
      "github_advisory"
    ],
    "scan_types": [
      "dependency_vulnerabilities",
      "code_analysis",
      "configuration_security",
      "secret_detection"
    ],
    "severity_levels": {
      "critical": "fail",
      "high": "warn",
      "medium": "info",
      "low": "ignore"
    },
    "exclusions": {
      "files": ["test_*.py", "mock_*.py"],
      "vulnerabilities": ["CVE-2021-44228"], // If false positive
      "paths": ["tests/", "examples/"]
    }
  }
}
```

### mcp-profiler Configuration
```json
{
  "mcp-profiler": {
    "profiling_modes": [
      "cpu_profiling",
      "memory_profiling",
      "io_profiling"
    ],
    "sampling_options": {
      "cpu_sample_rate": 1000,
      "memory_sample_rate": 100,
      "duration_limit": 300
    },
    "analysis_options": {
      "hotspot_detection": true,
      "call_graph_analysis": true,
      "memory_leak_detection": true
    },
    "output_formats": [
      "json",
      "flamegraph",
      "report"
    ]
  }
}
```

## 🚀 Usage Examples

### Basic Tool Usage
```python
# Check tool availability
def check_mcp_tools():
    """Check which MCP tools are available"""
    
    required_tools = ["mcp-filesystem", "mcp-git", "mcp-search"]
    optional_tools = ["mcp-analyzer", "mcp-profiler", "mcp-security"]
    
    available_tools = []
    missing_tools = []
    
    for tool in required_tools + optional_tools:
        if is_tool_available(tool):
            available_tools.append(tool)
        else:
            missing_tools.append(tool)
    
    return {
        "available": available_tools,
        "missing": missing_tools,
        "ready": all(tool in available_tools for tool in required_tools)
    }

# Execute tool with configuration
def execute_mcp_tool(tool_name, operation, config=None):
    """Execute MCP tool with specified configuration"""
    
    try:
        tool_config = load_tool_config(tool_name, config)
        result = tool_name.execute(operation, tool_config)
        return result
    except ToolNotAvailableError:
        return handle_tool_unavailable(tool_name)
    except TimeoutError:
        return handle_tool_timeout(tool_name, operation)
```

### Advanced Tool Orchestration
```python
# Parallel tool execution
async def execute_parallel_analysis():
    """Execute multiple tools in parallel"""
    
    analysis_tasks = [
        execute_mcp_tool("mcp-filesystem", "analyze_structure"),
        execute_mcp_tool("mcp-git", "history_analysis"),
        execute_mcp_tool("mcp-security", "vulnerability_scan")
    ]
    
    results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
    return process_parallel_results(results)

# Sequential tool execution with dependencies
def execute_sequential_analysis():
    """Execute tools in sequence with dependency management"""
    
    # Step 1: Filesystem analysis
    fs_result = execute_mcp_tool("mcp-filesystem", "scan")
    
    # Step 2: Git analysis (depends on filesystem)
    git_result = execute_mcp_tool("mcp-git", "analyze", {
        "target_files": fs_result.get("relevant_files")
    })
    
    # Step 3: Security scan (depends on both)
    security_result = execute_mcp_tool("mcp-security", "scan", {
        "target_files": fs_result.get("code_files"),
        "git_context": git_result.get("commit_info")
    })
    
    return combine_results([fs_result, git_result, security_result])
```

## 🔍 Troubleshooting Guide

### Common Issues

#### Tool Not Found
```bash
# Check tool installation
which mcp-filesystem
npm list -g @anthropic/mcp-tools

# Reinstall if necessary
npm uninstall -g @anthropic/mcp-tools
npm install -g @anthropic/mcp-tools

# Check PATH configuration
echo $PATH
```

#### Permission Errors
```bash
# Fix file permissions
chmod +x ~/.mcp/tools/*

# Fix directory permissions
chmod 755 ~/.mcp/

# Use sudo for system-wide installation
sudo npm install -g @anthropic/mcp-tools
```

#### Timeout Issues
```json
{
  "tools": {
    "mcp-profiler": {
      "timeout": 600000,  // Increase timeout
      "chunk_size": 1000,  // Process in smaller chunks
      "parallel_limit": 2  // Reduce parallel processing
    }
  }
}
```

#### Memory Issues
```json
{
  "performance": {
    "memory_limit": "4GB",
    "streaming_mode": true,
    "cache_size": "500MB"
  }
}
```

### Advanced Troubleshooting

#### Debug Mode
```bash
# Enable debug logging
export MCP_DEBUG=1

# Run with verbose output
mcp-analyzer --verbose --debug analyze /path/to/project

# Check debug logs
tail -f ~/.mcp/logs/debug.log
```

#### Tool Health Check
```python
def health_check_mcp_tools():
    """Comprehensive health check for MCP tools"""
    
    health_results = {}
    
    for tool_name in get_installed_tools():
        try:
            # Basic availability check
            is_available = check_tool_availability(tool_name)
            
            # Version check
            version = get_tool_version(tool_name)
            
            # Configuration validation
            config_valid = validate_tool_config(tool_name)
            
            # Performance test
            performance_ok = test_tool_performance(tool_name)
            
            health_results[tool_name] = {
                "available": is_available,
                "version": version,
                "config_valid": config_valid,
                "performance_ok": performance_ok,
                "status": "healthy" if all([is_available, config_valid, performance_ok]) else "issues"
            }
            
        except Exception as e:
            health_results[tool_name] = {
                "status": "error",
                "error": str(e)
            }
    
    return health_results
```

## ⚡ Performance Optimization

### Tool Performance Tuning
```json
{
  "performance": {
    "global": {
      "concurrent_tools": 3,
      "memory_limit": "4GB",
      "cache_enabled": true,
      "cache_size": "1GB"
    },
    "tool_specific": {
      "mcp-filesystem": {
        "parallel_file_processing": true,
        "chunk_size": 1000,
        "cache_file_metadata": true
      },
      "mcp-git": {
        "shallow_clone": true,
        "parallel_branch_analysis": true,
        "cache_git_objects": true
      },
      "mcp-profiler": {
        "sampling_optimization": true,
        "memory_efficient_mode": true,
        "result_streaming": true
      }
    }
  }
}
```

### Resource Management
```python
def optimize_tool_execution():
    """Optimize MCP tool execution for better performance"""
    
    # System resource monitoring
    system_resources = monitor_system_resources()
    
    # Dynamic configuration adjustment
    if system_resources["memory_available"] < "2GB":
        adjust_tool_config("memory_efficient_mode", True)
    
    if system_resources["cpu_cores"] > 4:
        adjust_tool_config("concurrent_tools", 4)
    
    # Load balancing
    distribute_tool_load(get_available_tools())
    
    # Caching optimization
    optimize_cache_usage(system_resources)
```

## 🔐 Security Configuration

### Security Best Practices
```json
{
  "security": {
    "access_control": {
      "restricted_paths": [
        "~/.ssh/",
        "~/.aws/",
        "/etc/",
        "/var/log/"
      ],
      "allowed_operations": [
        "read",
        "analyze"
      ],
      "forbidden_operations": [
        "delete",
        "modify_system_files"
      ]
    },
    "network_security": {
      "allow_internet_access": false,
      "trusted_domains": [
        "api.anthropic.com",
        "github.com"
      ],
      "proxy_settings": {
        "use_proxy": false,
        "proxy_url": ""
      }
    },
    "data_protection": {
      "encrypt_cache": true,
      "anonymize_logs": true,
      "secure_temp_files": true
    }
  }
}
```

### Audit Configuration
```json
{
  "audit": {
    "log_all_operations": true,
    "log_level": "INFO",
    "log_file": "~/.mcp/logs/audit.log",
    "log_rotation": {
      "max_size": "100MB",
      "max_files": 10
    },
    "sensitive_data_filtering": true,
    "compliance_reporting": true
  }
}
```

## 📊 Monitoring & Logging

### Logging Configuration
```json
{
  "logging": {
    "level": "INFO",
    "output": {
      "console": true,
      "file": true,
      "file_path": "~/.mcp/logs/mcp.log"
    },
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "rotation": {
      "max_size": "50MB",
      "backup_count": 5
    },
    "categories": {
      "performance": "DEBUG",
      "security": "WARNING",
      "errors": "ERROR"
    }
  }
}
```

### Performance Monitoring
```python
def setup_performance_monitoring():
    """Set up comprehensive performance monitoring"""
    
    monitoring_config = {
        "metrics": {
            "execution_time": True,
            "memory_usage": True,
            "cpu_usage": True,
            "tool_success_rate": True
        },
        "alerts": {
            "slow_execution": 300,  # seconds
            "high_memory": "1GB",
            "high_cpu": 80,  # percentage
            "failure_rate": 10  # percentage
        },
        "reporting": {
            "interval": 3600,  # seconds
            "format": "json",
            "destination": "~/.mcp/reports/"
        }
    }
    
    return configure_monitoring(monitoring_config)
```

## 🚀 Quick Start Checklist

### Initial Setup
- [ ] Install Cursor IDE and Claude.code extension
- [ ] Install required MCP tools (filesystem, git, search)
- [ ] Create global configuration file
- [ ] Verify tool availability and permissions
- [ ] Run health check

### Project Setup
- [ ] Create project-specific configuration
- [ ] Configure excluded files and paths
- [ ] Set quality thresholds
- [ ] Test tool execution with sample project
- [ ] Document project-specific settings

### Production Deployment
- [ ] Configure production-specific settings
- [ ] Set up monitoring and logging
- [ ] Configure security restrictions
- [ ] Set up automated backups
- [ ] Create rollback procedures

This comprehensive configuration guide ensures optimal setup and operation of MCP tools with Claude.code, providing both basic configuration for getting started and advanced options for production deployment.