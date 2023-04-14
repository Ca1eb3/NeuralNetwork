from setuptools import setup, find_packages

setup(
    name='deeplearning',
    version='1.1',
    packages=find_packages(include=["neuralnet", "neuralnetbuilder"]),
    install_requires=[
        'numpy',
    ]
)
