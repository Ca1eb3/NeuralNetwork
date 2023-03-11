from setuptools import setup, find_packages

setup(
    name='deeplearning',
    version='1.0',
    packages=find_packages(include=["neuralnet", "neuralnetbuilder"]),
    install_requires=[
        'numpy',
    ]
)
