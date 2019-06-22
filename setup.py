from setuptools import setup

requirements = open('requirements.txt').read().splitlines()

setup(name='Beamsearch_LM',
      description='Beamsearch With Language Model for CTC Decoding .',
      version='0.1.0',

      packages=['beamsearch_lm','beamsearch_lm.lm_models'],
      install_requires=requirements,
      include_package_data=True,
      zip_safe=False)
