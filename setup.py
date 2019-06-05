from setuptools import setup, find_packages


with open('README.md', 'r') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='spikelib',
    version='0.1.01',
    # scripts=['sta'],
    description='Set of tools to analyze neuronal activity from spiketimes',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Cesar Reyes',
    author_email='cesar.reyesp@gmail.com',
    url='https://github.com/creyesp/spikelib',
    classifiers=[
      'Intended Audience :: Science/Research',
      'Topic :: Scientific/Engineering :: Information Analysis',
      'Topic :: Scientific/Engineering :: Visualization',
      'License :: OSI Approved :: MIT License',
      'Operating System :: OS Independent',
      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 2.7'
    ],
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'numpy>=1.15.4',
        'scipy>=1.1.0',
        'matplotlib>=2.2.3',
        'scikit-learn>=0.20.1',
        'peakutils>=1.3.0',
        'pandas>=0.23.4',
        'h5py>=2.8.0',
        'lmfit>=0.9.12',
    ],
)
