import os
from setuptools import setup, find_packages

setup(
    name="melts",
    version="0.1.0",
    url="https://github.com/VillainNumberOne/multi-example-texture-synthesis",
    author="Maxim Gorshkov",
    author_email="nmvgorshkofff@gmail.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
    setup_requires=[
        "numpy>=1.19.5"
        "Pillow>=9.1.1",
        "requests>=2.25.1",
        "scikit_learn>=1.1.1",
        "setuptools>=49.2.1",
        "torch>=1.7.1+cu110",
        "torchvision>=0.8.2+cu110",
        "tqdm>=4.56.2"
    ],
    tests_require=[],
    extras_require={},
    include_package_data=True,
    package_data={'': ['models/*.pth']},
)