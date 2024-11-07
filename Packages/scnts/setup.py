from pathlib import Path
from setuptools import setup, find_packages

def read_requirements(path):
    return list(Path(path).read_text().splitlines())


reqs = read_requirements('requirements.txt')

setup(
    name = "SHATS",
    version = "0.1",
    packages = find_packages(),
    description = "A super new cool package for time series and forecasting",
    author = "Coolest authors",
    install_requires = reqs
)