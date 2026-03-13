from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    '''Method to get the list of requirements'''

    requirements = []
    with open(file_path, 'r') as file:
        requirements = file.readline().strip().split('\n')
        requirements = [req.replace("-n"," ") for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name="ml_pipeline",
    version="0.1",
    packages=find_packages(),
    author="afaque",
    author_email="afaquefarooq@gmail.com",
    install_requires=get_requirements('requirements.txt')
)