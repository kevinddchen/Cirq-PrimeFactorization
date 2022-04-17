from setuptools import setup, find_packages


setup(
    name="factor",
    version="0.1.0",
    python_requires=">=3.8",
    packages=find_packages("factor*"),
    install_requires=[
        "cirq==0.13.1",
    ],
    extras_require={
        "dev": ["pre-commit", "pytest"],
    },
)
