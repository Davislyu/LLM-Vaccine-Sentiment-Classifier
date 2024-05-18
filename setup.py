from setuptools import setup, find_packages

setup(
    name='twitter-classification',
    version='0.1',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'torch',
        'transformers',
        'tqdm',
        "scikit-learn"
    ],
)