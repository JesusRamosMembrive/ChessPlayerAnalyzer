# Migration Guide

This guide helps migrate from legacy documentation structure to the new organized format.

## Documentation Migration

The documentation has been completely reorganized from scattered markdown files to a structured format:

### Old Structure
```
├── README.md
├── CLAUDE.md
├── SYZYGY_SETUP.md
├── advanced-statistical-models.md
├── cross-game-correlation.md
├── enhanced-timing-patterns.md
├── positional-pattern-recognition.md
├── analysis-enhancement-plan.md
├── PERFORMANCE_ANALYSIS.md
└── docs/old/
    ├── apm_setup.md
    ├── backup_recovery.md
    ├── tasks.md
    ├── fairness.md
    └── suspicion_priors.md
```

### New Structure
```
docs/
├── README.md                 # Main documentation index
├── architecture/             # System architecture
├── api/                      # API documentation
├── modules/                  # Module documentation
├── guides/                   # Guides and procedures
├── algorithms/               # Algorithm documentation
└── legacy/                   # Legacy documentation
```

## File Mapping

| Old Location | New Location | Status |
|--------------|--------------|--------|
| `README.md` | `docs/guides/setup/README-original.md` | ✅ Migrated |
| `CLAUDE.md` | `docs/guides/setup/CLAUDE.md` | ✅ Migrated |
| `SYZYGY_SETUP.md` | `docs/guides/SYZYGY_SETUP.md` | ✅ Migrated |
| `advanced-statistical-models.md` | `docs/algorithms/advanced-statistical-models.md` | ✅ Migrated |
| `cross-game-correlation.md` | `docs/algorithms/cross-game-correlation.md` | ✅ Migrated |
| `enhanced-timing-patterns.md` | `docs/algorithms/enhanced-timing-patterns.md` | ✅ Migrated |
| `positional-pattern-recognition.md` | `docs/algorithms/positional-pattern-recognition.md` | ✅ Migrated |
| `analysis-enhancement-plan.md` | `docs/guides/analysis-enhancement-plan.md` | ✅ Migrated |
| `PERFORMANCE_ANALYSIS.md` | `docs/guides/PERFORMANCE_ANALYSIS.md` | ✅ Migrated |
| `docs/old/*` | `docs/legacy/old/` | ✅ Migrated |

## Link Updates

All internal links have been updated to reflect the new structure. If you have external links to the old documentation, update them as follows:

### Example Updates
```markdown
# Old links
[Setup Guide](README.md)
[Claude Setup](CLAUDE.md)
[Performance Analysis](PERFORMANCE_ANALYSIS.md)

# New links
[Setup Guide](guides/setup/README-original.md)
[Claude Setup](guides/setup/CLAUDE.md)
[Performance Analysis](guides/PERFORMANCE_ANALYSIS.md)
```

## Legacy Documentation

All legacy documentation is preserved in `docs/legacy/old/` for historical reference:

- `apm_setup.md` - APM monitoring setup (superseded by OpenTelemetry)
- `backup_recovery.md` - Backup procedures (updated in deployment guide)
- `tasks.md` - Task documentation (superseded by Celery workflow docs)
- `fairness.md` - Fairness analysis (integrated into analysis modules)
- `suspicion_priors.md` - Bayesian priors (integrated into Bayesian module docs)

## New Documentation Benefits

The new structure provides:

1. **Better Organization**: Logical grouping by functionality
2. **Improved Navigation**: Clear hierarchy and cross-references
3. **Enhanced Searchability**: Structured content for better discoverability
4. **Standardized Format**: Consistent documentation templates
5. **Maintenance**: Easier to keep documentation up-to-date

## References

- [Documentation Plan](../plan_de_accion.md) - Complete migration history
- [Documentation Conventions](../CONVENTIONS.md) - New documentation standards
- [Template Guide](../TEMPLATE.md) - Template for new documentation