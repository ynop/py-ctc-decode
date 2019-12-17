import os

from setuptools import find_packages
from setuptools import setup

##################################################
# Dependencies
##################################################

PYTEST_VERSION_ = '5.2.3'
KENLM_COMMIT = '96d303cfb1a0c21b8f060dbad640d7ab301c019a'

# Packages required in 'production'
REQUIRED = [
    'tqdm==4.39.0',
    'kenlm @ git+ssh://git@github.com/kpu/kenlm@{}#egg=kenlm'.format(
        KENLM_COMMIT
    ),
    'numpy==1.16.2',
    'psutil==5.6.7',
]

# Packages required for dev/ci enrionment
EXTRAS = {
    'dev': [
        'click==7.0',
        'pytest==%s' % (PYTEST_VERSION_,),
        'pytest-runner==5.2',
        'pytest-cov==2.8.1',
        'pytest-benchmark==3.2.2',
    ],
    'ci': [
        'flake8==3.7.9',
        'flake8-quotes==2.1.1'
    ],
}

# Packages required for testing
TESTS = [
    'pytest==%s' % (PYTEST_VERSION_,),
    'requests_mock==1.7.0'
]

##################################################
# Description
##################################################

DESCRIPTION = 'ctcdecode is package to decode output from CTC trained dnn.'

root = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(root, 'README.md')

# Import the README and use it as the long-description.
try:
    with open(readme_path, encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

##################################################
# SETUP
##################################################

setup(name='ctcdecode',
      version='0.0.0',
      description=DESCRIPTION,
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/ynop/ctcdecode',
      download_url='https://github.com/ynop/ctcdecode/releases',
      author='Matthias Buechi',
      author_email='buec@zhaw.ch',
      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3 :: Only',
          'Topic :: Scientific/Engineering :: Human Machine Interfaces'
      ],
      keywords=('speech recognition lexicon dictionary '
                'phone phoneme pronunciation'),
      license='MIT',
      packages=find_packages(exclude=['tests']),
      install_requires=REQUIRED,
      include_package_data=True,
      zip_safe=False,
      test_suite='tests',
      extras_require=EXTRAS,
      setup_requires=['pytest-runner'],
      tests_require=TESTS,
      entry_points={}
      )
