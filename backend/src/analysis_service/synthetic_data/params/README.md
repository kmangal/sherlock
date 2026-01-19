# Sampling Scenarios

This folder contains configuration files that are used to set parameters for synthetic exam generation under different scenarios.

Usage:

```python
from pathlib import Path
from analysis_service.synthetic_data.parameters import load_question_params
params = load_question_params(Path("scenarios/baseline.yaml"))
```

# TODO: Implement the following:
- bimodal
- difficult_exam
- easy_exam
- high_missingness
- large_scale
- skewed
- weak_distractors