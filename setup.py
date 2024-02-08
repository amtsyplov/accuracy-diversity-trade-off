import os

from setuptools import setup


def read_version() -> str:
    with open(os.path.abspath("build.number"), mode="r") as file:
        version = file.read()
    return version


if __name__ == "__main__":
    setup(
        name="divrec",
        version=read_version(),
        packages=[
            "divrec",
            "divrec.models",
            "divrec.datasets",
        ],
        url="",
        license="MIT License",
        author="amtsyplov",
        author_email="",
        description="",
        install_requires=list(),
    )
