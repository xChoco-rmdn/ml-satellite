from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    """
    Returns a list of requirements from requirements.txt
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements


setup(
    name='satellite-nowcasting',
    version='0.1.0',
    author='Rmdn',
    author_email='zulkiflirmdn@gmail.com',
    description='Satellite Nowcasting Project for NTB Region using Himawari-8/9 Data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Atmospheric Science :: Machine Learning :: Satellite Data',
    ],
    python_requires='>=3.8',
)
