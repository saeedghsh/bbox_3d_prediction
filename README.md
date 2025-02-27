[![black](https://github.com/saeedghsh/bbox_3d_prediction/actions/workflows/formatting.yml/badge.svg?branch=master)](https://github.com/saeedghsh/bbox_3d_prediction/actions/workflows/formatting.yml)
[![pylint](https://github.com/saeedghsh/bbox_3d_prediction/actions/workflows/pylint.yml/badge.svg?branch=master)](https://github.com/saeedghsh/bbox_3d_prediction/actions/workflows/pylint.yml)
[![mypy](https://github.com/saeedghsh/bbox_3d_prediction/actions/workflows/type-check.yml/badge.svg?branch=master)](https://github.com/saeedghsh/bbox_3d_prediction/actions/workflows/type-check.yml)
[![pytest](https://github.com/saeedghsh/bbox_3d_prediction/actions/workflows/pytest.yml/badge.svg?branch=master)](https://github.com/saeedghsh/bbox_3d_prediction/actions/workflows/pytest.yml)
[![pytest-cov](https://github.com/saeedghsh/bbox_3d_prediction/actions/workflows/pytest-cov.yml/badge.svg?branch=master)](https://github.com/saeedghsh/bbox_3d_prediction/actions/workflows/pytest-cov.yml)

# 3D Bounding Box Prediction
A deep learning pipeline for 3D bounding box prediction of objects from color image and point cloud data.

🚨 **Disclaimer: Experimental & Potentially Buggy**  
This project is still a work in progress and may contain bugs or unexpected behavior.

## Usage examples

<details>
<summary>Install dependencies</summary>

```bash
sudo apt install libgtk2.0-dev pkg-config
pip install -r requirements.txt
```
</details>

```bash
# Simple visualization of the dataset
python -m utils.visualize_data
```

## Development

<details>
<summary>Code quality checks</summary>

**Linter, formatter and type check**
```bash
$ black . --check
$ isort . --check-only
$ mypy .
$ pylint $(git ls-files '*.py')
```

**Tests and coverage**
```bash
$ pytest --cov=. --cov-report html; firefox htmlcov/index.html
$ coverage report -m # see which lines are missing coverage
```

NOTE: using `xvfb-run` is useful for running pytest when testing GUI stuff, e.g. on a headless remote (i.e. CI in `github/workflows`).
```bash
$ xvfb-run --auto-servernum pytest
```

**Profiling** (if needed)
```bash
python -m cProfile -o profile.out -m entry.script
tuna profile.out
```
</details>

<details>
<summary>TODO</summary>

**Workflow and code quality**:
* [ ] complete test coverage
* [ ] fix the discrepancy between local and remote execution of the `mypy`.
* [ ] speed up github CI workflows
</details>


## License
```
Copyright (C) Saeed Gholami Shahbandi
```

NOTE: Portions of this code/project were developed with the assistance of ChatGPT (a product of OpenAI) and Copilot (A product of Microsoft).

Distributed with a GNU GENERAL PUBLIC LICENSE; see [LICENSE](https://github.com/saeedghsh/bbox_3d_prediction/blob/master/LICENSE).