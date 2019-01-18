from codecs import open
import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md')) as f:
    long_description = f.read()

requirements = ['numpy>=1.13.0', 'tensorflow>=1.2']

setup(
    name='stemdl',
    version='0.1',
    description='',
    long_description=long_description,
    url='https://code.ornl.gov/disMultiABM',
    license='MIT',
    author='N. Laanait',
    author_email='laanaitn@ornl.gov',

    # I don't remember how to do this correctly!!!. NL
    install_requires=requirements,
    # package_data={'sample':['dataset_1.dat']}
    test_suite='nose.collector',
    tests_require='Nose',
    dependency='',
    dependency_links=[''],
    include_package_data=True

)
