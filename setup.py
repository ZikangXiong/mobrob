from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requires = f.read().splitlines()

setup(
    name="mobrob",
    version="0.0.1",
    description="Goal-conditioned control for mobile robot environments",
    author="Zikang Xiong",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=requires,
)
