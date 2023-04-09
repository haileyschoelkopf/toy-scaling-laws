import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="toy-scaling-laws",
    py_modules=["toy_scaling"],
    version="0.0.1",
    description="",
    author="Hailey Schoelkopf and Zhangir Azerbayev",
    packages=find_packages(),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    entry_points={
        "console_scripts": [
            "train = train",
        ]
    },
)