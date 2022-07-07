import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "opus",
    version = "0.0.1",
    author = "Opus Dev Team",
    description = ("Columnar data format optimized for images, audio, video, and sensor data"),
    classifiers=[
        "Development Status :: 3 - Alpha",        
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    license='Apache License, Version 2.0',
    url = "https://github.com/opusdata/opus",
    packages=['opus'],
    long_description=read('README.md'),
)
