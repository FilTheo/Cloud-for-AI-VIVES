from pathlib import Path
from setuptools import setup, find_packages

def read_requirements(path):
    return list(Path(path).read_text().splitlines())


reqs = read_requirements('requirements.txt')

setup(
    name = "scnts",
    version = "0.1",
    packages = find_packages(),
    description = "A super new cool package for time series and forecasting",
    author = "Coolest authors",
        install_requires = ["backports.zoneinfo==0.2.1",
                "cffi==1.15.0",
                "cycler==0.11.0",
                "Cython==0.29.24",
                "DateTime==4.3",
                "fonttools==4.28.2",
                "Jinja2==3.0.3",
                "joblib==1.1.0",
                "kiwisolver==1.3.2",
                "lightgbm==3.3.1",
                "MarkupSafe==2.0.1",
                "matplotlib==3.5.0",
                "numpy==1.21.4",
                "packaging==21.3",
                "pandas==1.3.4",
                "patsy==0.5.2",
                "Pillow==8.4.0",
                "pmdarima==1.8.4",
                "pycparser==2.21",
                "pyparsing==3.0.6",
                "python-dateutil==2.8.2",
                "pytz==2021.3",
                "pytz-deprecation-shim==0.1.0.post0",
                "rpy2==3.4.5",
                "scikit-learn==1.0.1",
                "scipy==1.7.3",
               "setuptools-scm==6.3.2",
                "six==1.16.0",
                "sklearn==0.0",
                "statsmodels==0.13.1",
                "threadpoolctl==3.0.0",
                "tomli==1.2.2",
                "tzdata==2021.5",
                "tzlocal==4.1",
                "urllib3==1.26.7",
                "xgboost==1.5.0",
                "zope.interface==5.4.0"
        ]
)
