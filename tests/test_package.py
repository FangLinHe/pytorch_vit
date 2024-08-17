from __future__ import annotations

import importlib.metadata

import pytorch_vit as m


def test_version():
    assert importlib.metadata.version("pytorch_vit") == m.__version__
