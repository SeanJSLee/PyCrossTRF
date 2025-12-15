from setuptools import setup, find_packages
import io

setup(
    name='pycrosstrf',
    version='0.1.0',
    author='Jaeseok Sean Lee',
    author_email='seanjslee@gmail.com',
    description='Python 3 package for Cross TRF function estimation',
    # long_description=open('README.md').read(),
    long_description=io.open('README.md', encoding='utf-8').read(), 
    long_description_content_type='text/markdown',
    url='https://github.com/SeanJSLee/PyCrossTRF',
    packages=find_packages(), # Automatically finds the package folder
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    # install_requires=[
    #     'somepackage>=1.0',
    #     'anotherpackage',
    # ],
)
