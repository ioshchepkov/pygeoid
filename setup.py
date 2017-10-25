#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages

import pygeoid

# Get the long description from the README file
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                       'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='pygeoid',
      version=pygeoid.__version__,
      description='Local gravity field modelling with Python',
      long_description=long_description,
      url='https://github.com/ioshchepkov/pygeoid',
      author='Ilya Oshchepkov',
      author_email='ilya.oshchepkov@gmail.com',
      license='MIT',
      classifiers=[
          'Development Status :: 1 - Planning',
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3 :: Only',
          'Topic :: Education',
          'Topic :: Scientific/Engineering',
      ],
      keywords=['geodesy', 'gravimetry', 'geoid'],
      packages=find_packages(),
      python_requires='>=3',
      )
