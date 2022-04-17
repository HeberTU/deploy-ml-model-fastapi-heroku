# -*- coding: utf-8 -*-
"""Module Test.

Created on: 4/17/2022
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
from src import __version__


def test_version():
    assert __version__ == "0.0.2"
