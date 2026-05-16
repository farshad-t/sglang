# Archived Expert Distribution Analysis Scripts

These are older versions of expert distribution analysis scripts, superseded by the latest `*_from_log.py` versions.

## Evolution Timeline

### Phase 1: Histogram-based approach (May 8, 2026)
- **analyze_expert_histogram_prefill.py**: Initial prefill analysis using histograms
- **rebucket_prefill_optimal.py**: Rebucketing optimization experiments
- **visualize_expert_histograms.py**: Visualization utilities

**Limitation**: Required manual specification of parameters, less automated

### Phase 2: Bucketed approach with directory input (May 10, morning)
- **analyze_expert_histogram.py**: Improved histogram generation
- **analyze_prefill_bucketed.py**: K-means bucketing on prefill data, takes directory input
- **analyze_decode_histogram.py**: Separate decode phase analysis

**Limitation**: Takes directory of .pt files as input, less explicit about which batches to analyze

### Phase 3: Log-based approach (May 10, evening) - **CURRENT**
- **analyze_prefill_from_log.py**: Takes log file(s) as input, extracts all metadata automatically
- **analyze_decode_from_log.py**: Decode analysis from log files

**Advantages**:
- More explicit: user specifies exact log files to analyze
- More reliable: extracts batch_size, input_len, output_len directly from logs
- Better multi-batch support: handles multiple log files naturally
- Cleaner workflow: one command analyzes specific benchmark runs

## Why Moved to Archive

The latest `*_from_log.py` scripts provide:
1. **Better usability**: Log files are what users naturally have after benchmarks
2. **More reliable metadata extraction**: No guessing or inference needed
3. **Clearer intent**: User explicitly chooses which runs to analyze
4. **Maintainability**: Single source of truth (log file) for all parameters

## If You Need to Use Old Scripts

These scripts still work but are not maintained. Refer to their internal documentation for usage.
For new analysis, always use the latest scripts in the parent directory.

Last archived: May 11, 2026
