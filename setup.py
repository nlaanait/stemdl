from setuptools import setup, find_packages



setup(
    name='stemdl',
    version='0.1',
    packages=['stemdl'],
    install_requires=['horovod','tensorflow', 'lmdb', 'scipy', 'numpy'],
    url='https://code.ornl.gov/disMultiABM',
    license='MIT',
    author='N. Laanait',
    author_email='laanaitn@ornl.gov',
    include_package_data=True

)
