from setuptools import setup

VERSION = '0.0.1' 
DESCRIPTION = 'Neural network framework built from scratch in Python'

setup(
    name='NeuralNetwork',
    url='https://github.com/sergiosg1504/NeuralNetwork.git',
    author='Sergio Sanchez and Pablo Santos',
    author_email='sergiosg@usal.es',
    packages=['neural-network'],
    install_requires=['numpy','pickleshare','matplotlib'],
    version=VERSION,
    license='MIT',
    description=DESCRIPTION,
    long_description=open('README.txt').read()
)