# Deprecated Prompts

This directory contains the original 8 prompts that have been consolidated into the new optimized collection.

## Migration Guide

### Old Prompt → New Prompt Mapping

| Original Prompt | New Consolidated Prompt | Mode/Section |
|----------------|-------------------------|--------------|
| `codebase_analysis_prompt.md` | `01_comprehensive_codebase_analysis.md` | Standard/Deep mode |
| `ultra_detailed_analysis_prompt.md` | `01_comprehensive_codebase_analysis.md` | Forensic mode |
| `ai_coder_todo_prompt.md` | `02_ai_development_completion.md` | All modes |
| `project_review_prompt.md` | `02_ai_development_completion.md` | Production/Enterprise mode |
| `codebase_cleanup_prompt.md` | `03_cleanup_optimization_engine.md` | Maintenance mode |
| `refactoring_optimization_prompt.md` | `03_cleanup_optimization_engine.md` | Performance mode |
| `file_cleanup_prompt.md` | `03_cleanup_optimization_engine.md` | Maintenance mode |
| `markdown_documentation_generator.md` | `01_comprehensive_codebase_analysis.md` | Documentation generation |

### Key Improvements in New Version

1. **Reduced Redundancy**: Eliminated 60-80% overlap between prompts
2. **Better Organization**: Clear mode selection instead of multiple similar prompts
3. **Practical Examples**: Added real-world usage examples and scenarios
4. **Error Handling**: Comprehensive error handling and recovery strategies
5. **Safety Features**: Built-in safety checks and rollback capabilities
6. **Performance**: Optimized for better resource usage and execution speed

### Migration Steps

1. **Identify Current Usage**: Determine which old prompts you're currently using
2. **Select New Prompt**: Choose the appropriate new prompt and mode
3. **Update Configuration**: Migrate any custom configuration to new format
4. **Test Execution**: Verify new prompt works with your workflow
5. **Update Documentation**: Update any internal documentation references

### Backward Compatibility

These old prompts are deprecated but remain available for reference. However, they will not receive updates or bug fixes. Please migrate to the new consolidated prompts at your earliest convenience.

## Support

For migration assistance or questions about the new prompts, refer to:
- `00_prompt_index.md` - Complete usage guide
- `04_mcp_tool_configuration.md` - Configuration instructions
- GitHub Issues - For specific migration problems

---

**Note**: These deprecated prompts will be removed in a future version. Please migrate to the new consolidated prompts.