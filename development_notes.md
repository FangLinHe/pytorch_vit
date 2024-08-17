# Notes

## Create Project
* Install and activate virtual environment

```
$ pyenv virtualenv 3.12.3 torch_vit
$ pyenv activate torch_vit
```

* Install dependency [Cookiecutter](https://github.com/cookiecutter/cookiecutter)
  ```
  pip install pipx
  pipx install cookiecutter
  ```
* Use Cookiecutter template
  ```
  cookiecutter gh:dirmeier/cookiecutter-python-ml-project
  ```
