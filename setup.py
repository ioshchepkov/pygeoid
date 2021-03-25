#! /usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

import versioneer

with open('README.rst') as readme_file:
    long_description = readme_file.read()

name = 'pygeoid'
version = versioneer.get_version()
cmdclass = versioneer.get_cmdclass()

install_requires = ['numpy', 'scipy', 'pyproj',
                    'astropy', 'pyshtools', 'joblib'],

setup(name=name,
      version=version,
      cmdclass=cmdclass,
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
          'Programming Language :: Python :: 3 :: Only',
          'Programming Language :: Python :: 3.9',
          'Topic :: Education',
          'Topic :: Scientific/Engineering',
      ],
      keywords=['geodesy', 'gravimetry', 'geoid', 'gravity'],
      packages=find_packages(),
      include_package_data=True,
      tests_require=['pytest'],
      setup_requires=['pytest-runner'],
      install_requires=install_requires,
      python_requires='>=3.9')
