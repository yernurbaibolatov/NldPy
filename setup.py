from setuptools import setup, find_packages

setup(
    name="nldpy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "matplotlib"],
    description="Numerical analysis of nonlinear dynamical systems",
    author="Yernur Baibolatov",
    author_email="yernurb@gmail.com",
    url="https://github.com/yernurbaibolatov/NldPy",
)