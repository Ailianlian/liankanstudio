#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

""" liankanstudio setup.py

To install liankanstudio:

    python setup.py install

To run the liankanstudio unit tests (when they will exists...):

    python setup.py test
"""

# Standard Library Imports
import sys
from setuptools import setup
from glob import glob
import os

def get_requires():
    # type: () -> List[str]
    """ Get Requires: Returns a list of required packages. """
    return [
        'Click',
        'numpy',
        'tqdm'
    ]

def get_extra_requires():
    # type: () -> Dict[str, List[str]]
    """ Get Extra Requires: Returns a list of extra/optional packages. """
    return {
        # TODO: Abstract this into a function that generates this
        # dictionary based on a list of compatible Python & opencv-python
        # package versions (will need to use the output for requirements.txt).
        # TODO: Is there a tool that can do this automagically?
        'opencv:python_version <= "3.5"':
            ['opencv-contrib-python<=4.2.0.32'],
        'opencv:python_version > "3.5"':
            ['opencv-contrib-python==4.4.0'],

        'opencv-headless:python_version <= "3.5"':
            ['opencv-python-headless<=4.2.0.32'],
        'opencv-headless:python_version > "3.5"':
            ['opencv-python-headless'],
    }

# we are going to use 0.0.0.0 for version
# w.x.y.z -> w+1.0.0.0 for very "big change"
# w.x.y.z -> w.x+1.0.0 for every features
# w.x.y.z -> w.x.y+1.0 for every fix/bug in a release (useless in fact :D)
# w.x.y.z -> w.x.y.z+1 for every new dev improvment/test done.
setup(
    name='liankanstudio',
    version='0.4.0.5',
    description="Wonderful answer to wonderful exercise.",
    long_description=open(os.path.join('package_info.rst')).read(),
    author='Paulien Jeunesse',
    author_email='jeunesse.paulien@gmail.com',
    license="MIT",
    keywords="video computer-vision analysis",
    install_requires=get_requires(),
    extras_require=get_extra_requires(),
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    package_dir={"liankanstudio":"src/liankanstudio"},
    packages=['liankanstudio',
              'liankanstudio.util',
              'liankanstudio.data',
              'liankanstudio.data.ssd',
              'liankanstudio.data.yolo'],
    package_data = {
    '': [ '*.prototxt','*.caffemodel','*.cfg','*.weights' ],
},
    #data_files=[('.', ['../README.md','../package_info.rst']),
    #            ('dnn/ssd',glob('../data/ssd/*'))],
    #include_package_data = True,           # Must leave this to the default.
    #test_suite="unitest.py",               # Auto-detects tests from setup.cfg
    entry_points={"console_scripts": ["liankan=liankanstudio.cli:liankan"]},
)