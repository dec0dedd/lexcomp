from setuptools import setup
import os
import re

project_name = "LexComp"
version_file = os.environ["VERSION_FILE"]


def long_description():
    with open('README.md') as fp:
        return fp.read()


def get_requirements():
    with open("requirements.txt") as fp:
        dependencies = (d.strip() for d in fp.read().split("\n") if d.strip())
        return [d for d in dependencies if not d.startswith("#")]

setup(
    name=project_name,
    version='0.0.1',
    description="Python library for text complexity analysis",
    long_description=long_description(),
    long_description_content_type="text/markdown",
    author="Michał Burzyński",
    author_email="michal.burzynski0805@gmail.com",
    license="LGPL-3.0-ONLY",
    url="https://github.com/dec0dedd/lexcomp",
    packages=['lexcomp'],
    python_requires="> 3.10",
    install_requires=get_requirements()
)
