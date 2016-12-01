from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='feainterp',

    version='',

    description='Interpolate values at points inside of elements for finite '
                + 'element analysis (FEA).',
    long_description=long_description,

    url='https://github.com/tingwai-to/FEA_interpolation',

    author='Ting-Wai To',
    author_email='',

    license='',

    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Mathematics'

        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
    ],

    keywords='mathematics',

    packages=find_packages(exclude=['tests']),

    install_requires=[
        'numpy',
        'scipy',
        'numba',
    ],

    package_data={
        'examples': ['test_data.py'],
    },

    # entry_points={
    #     'console_scripts': [
    #         'feainterp=jit_setup',
    #     ],
    # },
)