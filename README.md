# beamsearch_lm
Beam-search with Language Model for CTC Decoding

## Installation
Install package
```bash
pip install git+ssh://git@github.com/phvan2312/beamsearch_lm.git
```

## Usage
The model uses CPU by default. No support for runtime changes for now. To run inference
```python
from beamsearch_lm.model_wrapper import SklearnAutoCorrectWrapper

lm_model = joblib.load("where_is_your_saving_path")

# remember input of this model is type numpy.ndarray, shape of (time_step, n_vocab)
input = "ctc_logit_matrix"
predict = lm_model.predict(input)

print (predict")
```

## Uninstallation
```bash
pip uninstall beamsearch-lm
```
