from setuptools import setup, find_packages

REQUIRED_PKGS = [
    "import-ipynb==0.1.3",
    "torch>=1.4.0",
    "transformers>=4.18.0",
    "numpy>=1.18.2",
    "accelerate",
    "datasets>=2.8.0",
    "statsmodels>=0.13.5",
    "scikit-learn>=1.2.2",
    "jsonlines>=3.1.0",
    "python-dotenv>=1.0.0",
]

setup(
    name='chattransformer',
    version='0.1.0',
    packages=find_packages(include=['chattransformer', 'chattransformer.*']),
    install_requires=REQUIRED_PKGS,
    extras_require={
        'interactive': ['matplotlib>=2.2.0', 'jupyter'],
    }
)