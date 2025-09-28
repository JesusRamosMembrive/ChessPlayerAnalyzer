# Test Script Strategy - Cumulative Approach

## Recommended Structure

```bash
#!/bin/bash
# test_full_flow.sh - Comprehensive test of all functionality

set -e  # Exit on any error

echo "🧪 Starting Full Flow Test..."

# =============================================================================
# PHASE 1: Basic Functionality (implemented: DD/MM/YYYY)
# =============================================================================
echo "📋 Testing Phase 1: Basic Setup..."

# Test basic functionality here
echo "✅ Phase 1 passed"

# =============================================================================
# PHASE 2: Data Persistence (implemented: DD/MM/YYYY)
# =============================================================================
echo "📋 Testing Phase 2: Data Persistence..."

# Build on Phase 1 + test new functionality
echo "✅ Phase 2 passed"

# =============================================================================
# PHASE 3: User Authentication (implemented: DD/MM/YYYY)
# =============================================================================
echo "📋 Testing Phase 3: Authentication..."

# Build on previous phases + test auth
echo "✅ Phase 3 passed"

# =============================================================================
# CLEANUP
# =============================================================================
echo "🧹 Cleaning up test data..."
# Clean any test artifacts

echo "🎉 All tests passed! System working correctly."
```

## Best Practices

### DO:
✅ **Organize by phases/features** with clear comments
✅ **Add timestamps** when each section was implemented
✅ **Use descriptive echo statements** to track progress
✅ **Include cleanup section** to reset state
✅ **Exit on first failure** (`set -e`) to pinpoint issues
✅ **Keep each section focused** but comprehensive
✅ **Add brief comments** explaining complex test logic

### DON'T:
❌ Let it become one giant unorganized blob
❌ Mix unit tests with integration tests
❌ Make tests depend on previous test artifacts (clean state each time)
❌ Skip validation of previous functionality when adding new sections
❌ Make it longer than 10-15 minutes execution time

## Hybrid Approach (Advanced)

If the script gets too long, consider this structure:

```
tests/
├── test_full_flow.sh          # Main comprehensive test
├── phase_tests/
│   ├── test_phase1.sh         # Individual phase tests
│   ├── test_phase2.sh         # Can be run in isolation
│   └── test_phase3.sh
└── utils/
    ├── test_helpers.sh        # Common test utilities
    └── cleanup.sh             # Shared cleanup functions
```

## Prompt for Claude Code

```
When creating or updating tests, please:

1. **Add to the existing test_full_flow.sh** rather than creating new files
2. **Organize the new test as a new phase section** with clear headers
3. **Ensure it builds on previous phases** and validates the complete flow
4. **Include descriptive output** so I can see progress and pinpoint failures
5. **Add cleanup for any new test data** created
6. **Keep each test section focused** but comprehensive

The goal is one script that validates the entire system works end-to-end, organized by development phases.
```

## When to Split

Consider separate scripts only when:
- Main script takes more than 10-15 minutes
- You have truly independent modules that don't interact
- You need to run tests in different environments
- Performance testing requires different setup

## Example Section Template

```bash
# =============================================================================
# PHASE X: [Feature Name] (implemented: [date])
# =============================================================================
echo "📋 Testing Phase X: [Feature Name]..."

# 1. Setup for this phase (if needed)
echo "  Setting up [feature] test..."

# 2. Test the new functionality
echo "  Testing [specific capability]..."
[test commands]

# 3. Verify integration with previous phases
echo "  Verifying integration..."
[integration test commands]

# 4. Validate end-to-end flow still works
echo "  Testing complete flow..."
[e2e validation]

echo "✅ Phase X passed"
```