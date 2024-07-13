from setuptools import setup, find_packages


setup(
    name='torch_sla',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=1.8.0',
        'ninja'
    ],
    extras_require={
        'test':['pytest','numpy','scipy'],
        'docs':['sphinx']
    }
)