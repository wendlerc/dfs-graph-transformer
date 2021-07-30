import setuptools
from setuptools import setup

#python setup.py build
#python setup.py install

__version__ = "0.0.1"


setup(
    name="dfs_transformer",
    version=__version__,
    author="Chris Wendler",
    author_email="chris.wendler@inf.ethz.ch",
    description="Transformers operating on minimal DFS codes.",
    long_description="",
    packages=setuptools.find_packages(where = 'src'),
    package_dir = {"":"src"},
    include_package_data=True,
)
