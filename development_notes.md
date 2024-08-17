# Notes

## Create Project
* Install and activate virtual environment

```
$ pyenv virtualenv 3.12.3 torch_vit
$ pyenv activate torch_vit
```

* Install [Cookiecutter](https://github.com/cookiecutter/cookiecutter)
  ```
  pip install pipx
  pipx install cookiecutter
  ```
* Use Cookiecutter template
  ```
  cookiecutter gh:scientific-python/cookie
  ```
* Install dependencies for development
  ```
  pip install -r requirements.txt
  ```
* Install pre-commit hook
  ```
  pre-commit install
  ```
* Run pre-commit hook manually
  ```
  pre-commit run --all-files
  ```
* Install package and run test
  ```
  pip install .
  pytest
  ```
