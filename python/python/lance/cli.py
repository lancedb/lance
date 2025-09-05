from setuptools import setup
from .lance import _lance_tools_cli
import sys

def lance_tools_cli():
    _lance_tools_cli(sys.argv)

