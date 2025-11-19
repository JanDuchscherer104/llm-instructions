# Copilot Instructions — Multivariate Data Analysis (ama-tlbx)

This document orients AI coding agents to the **ama-tlbx** codebase: architecture, usage patterns, and extension points.

---

## 1. Quick Start (TL;DR)

### Environment & Tools
- **Stack:** pandas, numpy, scikit‑learn, matplotlib/seaborn, pytest,
- **Style:** typed Python, Google docstrings, method‑chaining, pure analyzers, immutable views, strong separation of concerns
- **Python:** Conda environment `ama` with Python 3.13.9 at `/opt/homebrew/Caskroom/miniconda/base/envs/ama/bin/python`

### Important MCP Tools
- **Documentation:** `#get-library-docs` with library IDs:
  - Pandas: `pandas.pydata.org/docs`
  - scikit-learn: `scikit-learn.org/stable`
  - seaborn: `/mwaskom/seaborn`
- **Python Analysis:** `#pylance*` tools for context, refactoring, and code checks
- **Notebooks:** `#runNotebooks` to execute cells and validate code

### Make Commands
Run from project root:
```bash
make help               # List all available commands
make lint format fix    # Code quality tools
make check              # Run type checking
make test test-cov      # Run tests with coverage
make context-*          # Generate architecture diagrams (classes, packages, directory tree)
make docs-*             # Build quarto and pdoc documentation
```

### Agentic Behaviors
- **Always** follow code conventions (type hints, Google docstrings, Column enums)
- **Prefer** using existing external and internal functionalities (analyzers, plotting functions) - when confronted with a new requirement or problem, `#think` (mcp) about existing internal or external code. We can always add new libraries! Feel free to test code ideas in the terminal before proceeding.
- **Follow** established patterns when adding features (factories, immutable views, pure analyzers)
- **Ensure** code quality:
  - **For notebooks:** Add debug/dev cells (labeled with `# dev`) to print relevant information for understanding and debugging. Always execute cells and verify they run without errors before proceeding to subsequent cells.
  - **For package code:** Run `ruff format <file>` -> `ruff check <file>` -> `pytest <file>` to ensure code style compliance after finishing work on a file; Work test-driven and run pytest for changed code.
   Execute `make ci` for changes affecting multiple files to validate the full continuous integration pipeline before terminating.
- **Use** your MCP tools to retrieve helpful context, and utilize `make context-*` to understand codebase structure before major changes and when getting started
  - `context-package` to summarize symbols per module (classes/functions/constants),
  - `context-classes` to list classes with docstrings (first paragraph)
  - `context-classes-full-doc` to list classes with full docstrings - **Most Useful Starting Point**
- **Always** inspect all referenced symbols or files and get an initial understanding, then `#think` to plan your next steps before potentially gathering additional context or making code changes.

### Code Practices and Conventions (must follow)

1. **Always use Column enums for data access** — `df[LECol.GDP]` not `df["gdp"]` (type safety + IDE autocomplete)
2. **Use factory methods over direct instantiation** — `dataset.make_pca_analyzer()` ensures proper view configuration
3. **Pandas Practices**
    — Chain pandas operation to improves readability and performance; use `.query()`, `.assign()`, `.pipe()`
    - Use named aggregation in groupby for clear outputs
    - Use vectorized operations over row-wise `apply()`; avoid loops
4. **Type hint everything** — Function signatures, return types, variables; use `from typing import TYPE_CHECKING` for expensive imports
5. **Google-style docstrings with examples** — Include theory/assumptions, mathematical formulas, usage examples with `>>>`
6. **Analyzers are pure** — No plotting, no side effects, just computation → `.fit()` → `.result()`
7. **Plotting functions accept `*Result` objects** — Never raw DataFrames; ensures all metadata (pretty names) is available
8. **Views are immutable** — `DatasetView` is frozen; analyzers cannot accidentally modify source data
9. **Interpretation** - Always provide context and interpretation for analysis results in notebooks - ensure to inspect the outputs in a manner that allows you to draw meaningful conclusions and insights from the data and provide these interpretations alongside the results.

---

## 2. Architecture Overview

### Three-Layer Design: Separation of Concerns

The package follows strict separation between data handling, computation, and visualization:

```
ama_tlbx/
├── data/         # I/O, column definitions, immutable views
├── analysis/     # Pure computation (no plotting, no side effects)
└── plotting/     # Visualization only (accepts *Result objects)
```

### Core Components

#### `BaseDataset`
- Loads/cleans data and builds **DatasetView** snapshots
- Exposes factory methods for analyzers
- Every dataset (e.g., `LifeExpectancyDataset`) subclasses this

#### `DatasetView` *(immutable)*
- The **only** interface analyzers see (prevents accidental data modification)

```python
@dataclass(frozen=True)
class DatasetView:
    df: pd.DataFrame
    pretty_by_col: Mapping[str, str]
    numeric_cols: list[str]
    target_col: str | None = None
```

#### Analyzers *(pure computations)*
- `CorrelationAnalyzer`, `PCAAnalyzer`, `OutlierAnalyzer`, etc.
- No plotting, no side effects — just `.fit()` → `.result()`
- Operate exclusively on `DatasetView` objects and return `*Result` dataclasses

#### Plotting Functions
- Accept `*Result` objects (never raw DataFrames)
- Return `matplotlib.figure.Figure` objects
- Leverage pretty names from Result metadata

### Design Patterns

#### Strategy Pattern — Outlier Detection
```python
class OutlierDetector(ABC):
    @abstractmethod
    def detect(self, data: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame: ...

# Concrete strategies:
IQROutlierDetector(threshold=1.5)
ZScoreOutlierDetector(threshold=3.0)
IsolationForestOutlierDetector(contamination="auto")
```

#### Factory Pattern — Analyzer Creation
```python
dataset = LifeExpectancyDataset.from_csv() # Load dataset
corr_analyzer = dataset.make_correlation_analyzer(standardized=True, include_target=False)
pca_analyzer = dataset.make_pca_analyzer(standardized=True, exclude_target=True)
```

---

## 3. Usage Guide

### Quick Recipes

#### Load → Correlate → Plot
```python
le_ds = LifeExpectancyDataset.from_csv() # loads default CSV
corr_res = le_ds.make_correlation_analyzer(standardized=True, include_target=False).fit()
fig = plot_correlation_heatmap(corr_res)  # returns matplotlib.figure.Figure
```

#### PCA in one Line
```python
pca_result = LifeExpectancyDataset.from_csv().make_pca_analyzer(standardized=True, exclude_target=True).fit(n_components=None).result()
```

#### Outlier Detection with Strategy Pattern
```python
dataset = LifeExpectancyDataset.from_csv()
iqr_detector = IQROutlierDetector(threshold=1.5)
outliers = dataset.detect_outliers(iqr_detector, standardized=True)
```

---

## 4. Methods Reference

### Data Layer — `BaseDataset` & `LifeExpectancyDataset`

#### Loading & Properties
```python
# Factory method (class method)
LifeExpectancyDataset.from_csv(csv_path=None, aggregate_by_country=True, drop_missing_target=True)

# Core properties
dataset.df                   # Raw/cleaned DataFrame
dataset.df_standardized      # Standardized (mean=0, std=1) numeric features
dataset.df_pretty            # DataFrame with pretty column names
dataset.numeric_cols         # Index of numeric column names
```

#### View Creation
```python
# Generic view builder
dataset.view(columns=None, standardized=False, target_col=None) -> DatasetView

# Analyzer-optimized view (excludes identifiers by default)
dataset.analyzer_view(columns=None, standardized=True, include_target=True) -> DatasetView

# Get feature columns with flexible exclusions
dataset.feature_columns(include_target=False, extra_exclude=None) -> list[str]
```

#### Analyzer Factories
```python
# Correlation analyzer
dataset.make_correlation_analyzer(columns=None, standardized=True, include_target=True) -> CorrelationAnalyzer

# PCA analyzer
dataset.make_pca_analyzer(columns=None, standardized=True, exclude_target=True) -> PCAAnalyzer

# Outlier detection (strategy pattern)
dataset.detect_outliers(detector, columns=None, standardized=True) -> pd.DataFrame
```

#### Column Name Helpers
```python
dataset.get_pretty_name(col_name: str) -> str            # Single column
dataset.get_pretty_names(col_names: list[str]) -> list[str]  # Multiple columns
```

---

### Analysis Layer — Analyzers

#### `CorrelationAnalyzer`
```python
# Initialization (prefer factory method)
analyzer = dataset.make_correlation_analyzer(standardized=True, include_target=True)

# Core methods
analyzer.get_correlation_matrix() -> pd.DataFrame                # Full Pearson correlation matrix
analyzer.get_top_correlated_pairs(n=20) -> pd.DataFrame          # Top absolute correlations
analyzer.get_target_correlations() -> pd.DataFrame               # Feature-target correlations

# Result packaging
analyzer.fit(top_n_pairs=20) -> CorrelationResult
```

#### `PCAAnalyzer`
```python
# Initialization (prefer factory method)
analyzer = dataset.make_pca_analyzer(standardized=True, exclude_target=True)

# Fitting & transformation
analyzer.fit(n_components=None, exclude_cols=None) -> PCAAnalyzer  # Fit PCA model
analyzer.transform(n_components=None) -> pd.DataFrame              # Project to PC space

# Component analysis
analyzer.get_explained_variance() -> pd.DataFrame                  # Variance per component
analyzer.get_loading_vectors(component=None) -> pd.DataFrame       # Feature loadings
analyzer.get_top_loading_features(n_components=3, method="sum") -> pd.Index

# Result packaging
analyzer.result() -> PCAResult
```

#### `OutlierDetector` (Strategy Pattern)
```python
# Three concrete strategies
IQROutlierDetector(threshold=1.5)                    # Interquartile range rule
ZScoreOutlierDetector(threshold=3.0)                 # Standardized score thresholding
IsolationForestOutlierDetector(contamination="auto") # Ensemble anomaly detection

# Unified interface
detector.detect(data: pd.DataFrame, columns: Iterable[str] | None) -> pd.DataFrame  # Boolean mask
```

---

### Plotting Layer — Visualization Functions

#### Correlation Plots (`plotting/correlation_plots.py`)
```python
plot_correlation_heatmap(result: CorrelationResult, figsize=(20,20), **kwargs) -> Figure
plot_top_correlated_pairs(result: CorrelationResult, n=20, figsize=(12,10)) -> tuple[Figure, Figure]
plot_target_correlations(result: CorrelationResult, n=10, figsize=(10,6)) -> tuple[Figure, Figure]
```

#### PCA Plots (`plotting/pca_plots.py`)
```python
plot_explained_variance(result: PCAResult, figsize=(10,6)) -> Figure
plot_loadings_heatmap(result: PCAResult, n_components=3, top_n_features=None, figsize=(12,6)) -> Figure
```

#### Dataset Plots (`plotting/dataset_plots.py`)
```python
plot_standardization_comparison(dataset: BaseDataset, figsize=(20,10)) -> Figure
```

---

## 5. Extension Guide

### Adding a New Analyzer

**1. Create analyzer class** (in `analysis/my_analyzer.py`):
```python
from dataclasses import dataclass
from ama_tlbx.data.views import DatasetView

@dataclass(frozen=True)
class MyAnalysisResult:
    """Results package for MyAnalyzer."""
    summary: pd.DataFrame
    # ... other result fields ...

class MyAnalyzer:
    """Pure computation analyzer (no plotting!)."""

    def __init__(self, view: DatasetView):
        self._view = view
        self._fitted = False

    def fit(self) -> "MyAnalyzer":
        """Compute the analysis."""
        # ... computation logic ...
        self._fitted = True
        return self

    def result(self) -> MyAnalysisResult:
        """Return packaged results."""
        if not self._fitted:
            raise ValueError("Call fit() first")
        return MyAnalysisResult(...)
```

**2. Add factory method** to `BaseDataset`:
```python
def my_analyzer(
    self,
    *,
    columns: Iterable[str] | None = None,
    standardized: bool = True,
) -> MyAnalyzer:
    """Instantiate MyAnalyzer for this dataset."""
    from ama_tlbx.analysis.my_analyzer import MyAnalyzer
    view = self.analyzer_view(columns=columns, standardized=standardized)
    return MyAnalyzer(view)
```

### Adding Visualization Functions

**In `plotting/my_plots.py`**:
```python
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from ama_tlbx.analysis.my_analyzer import MyAnalysisResult

def plot_my_analysis(
    result: MyAnalysisResult,
    figsize: tuple[int, int] = (10, 6),
    **kwargs,
) -> Figure:
    """Plot results from MyAnalyzer.

    Args:
        result: Analysis results from MyAnalyzer.fit().result()
        figsize: Figure dimensions in inches.
        **kwargs: Additional keyword arguments passed to plotting function.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    # ... plotting logic using result ...
    fig.tight_layout()
    return fig
```

**Key principles:**
- Accept `*Result` dataclasses (ensures metadata like pretty names are available)
- Return `Figure` objects (not tuples unless multiple axes needed)
- Accept `**kwargs` to forward to underlying plot functions

---

## 6. Course Context: Multivariate Data Analysis Module

### Research Requirements
- Get research question approved by instructor before deep work

### Required Analysis Flow
1. **Preprocessing:** cleaning, recoding, missing value handling
2. **Descriptives:** visualize key variables (especially the **target**)
3. **EDA:** **PCA and clustering** to explore structure
4. **Regression:** linear or logistic with:
   - **Model choice:** feature selection, (non)linearity, **interactions**, **AIC**
   - **Assumption checks:** multicollinearity, residuals, normality
   - **Predictive performance:** CV / bootstrap / train–test split
   - **Interpretation:** effect sizes, p-values, R²
   - **Answer:** which features matter and how well can we predict?

### Code Quality
- Clean, structured, documented, reproducible
- Follow ama-tlbx conventions and patterns

### Optional Bonus
- Compare **ML model** (e.g., trees / random forest / boosting) against regression baseline
