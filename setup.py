from setuptools import setup

setup(
    name='stellar',
    version='2.0.0',
    author='Olive Stam, Cameron Ross, Austin Martinez',
    author_email='osstam@cpp.edu, ceross@cpp.edu, austinm@cpp.edu',
    description= 'A utility package for CPP Fire Squad',
    license='MIT',
    url='https://github.com/CPP-FIRE-Squad/stellar',
    py_modules=['Halo', 'Simulation', 'Particle'],
    install_requires=[
        'astropy'
    ],
)
