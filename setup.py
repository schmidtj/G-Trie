from distutils.core import setup

with open('README.txt') as file:
    long_description = file.read()
    
setup(
    name='GTrie',
    version='0.0.1',
    description='A python implementation of G-Trie',
    long_description=long_description,
    author='Jeffrey Schmidt',
    author_email='jschmid1@binghamton.edu',
    license = 'BSD',
    url='',
    packages=['GTrie']
      )
