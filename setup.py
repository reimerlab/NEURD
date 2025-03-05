from os import path
from pathlib import Path
from setuptools import setup, find_packages
from typing import List

here = path.abspath(path.dirname(__file__))

def get_install_requires(filepath=None):
    if filepath is None:
        filepath = "./"
    """Returns requirements.txt parsed to a list"""
    fname = Path(filepath).parent / 'requirements.txt'
    targets = []
    if fname.exists():
        with open(fname, 'r') as f:
            targets = f.read().splitlines()
            
    targets += get_links()
    return targets

def get_links():
    return [
        #"datasci_tools @ git+https://github.com/bacelii/datasci_tools.git'"
    ]

def get_long_description(filepath='README.md'):
    try:
        import pypandoc
        long_description = pypandoc.convert_file(filepath, 'rst') 
    except:
        print("\n\n\n****Need to install pypandoc (and if havent done so install apt-get install pandoc) to make long description clean****\n\n\n")
        
        long_description = Path("README.md").read_text()
        
    return long_description

# read in version number into __version__
with open(path.join(here, 'neurd', 'version.py')) as f:
    exec(f.read())

setup(
    name='neurd', # the name of the package, which can be different than the folder when using pip instal
    version=__version__,
    description='A mesh decomposition framework for automated proofreading and morphological analysis of neuronal EM reconstructions',
    long_description=get_long_description(),
	project_urls={
	    'Source':"https://github.com/reimerlab/NEURD/",
	    'Documentation':"https://reimerlab.github.io/NEURD/",
	},
	author='Brendan Celii',
	author_email='brendanacelii@gmail.com',
    packages=find_packages(),  #teslls what packages to be included for the install
    # package_data = {
    #     'neurd':['neurd/model_data/*'],
    # },
    include_package_data=True,
    install_requires=get_install_requires(), #external packages as dependencies
    # dependency_links = get_links(),
    # if wanted to install with the extra requirements use pip install -e ".[interactive]"
    extras_require={
        #'interactive': ['matplotlib>=2.2.0', 'jupyter'],
    },
    
    # if have a python script that wants to be run from the command line
    entry_points={
        #'console_scripts': ['pipeline_download=Applications.Eleox_Data_Fetch.Eleox_Data_Fetcher_vp1:main']
    },
    scripts=[], 
    
)

