from setuptools import setup, find_packages

setup(
    name='flexibletopology',
    version='0.1',
    py_modules=['flexibletopology'],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
    ],
)
