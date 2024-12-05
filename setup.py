from setuptools import setup, find_packages

from pathlib import Path


setup(
    name="HSI_Pocessing",
    version="0.1.0",
    author="Pukhkii Konstantin",
    author_email="konstantin.os.1204@gmail.com",
    description="Methods for processing hyperspectral data.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Kosty1024Bit/HSI-Processing",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache-2.0 license",  # Укажите вашу лицензию
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=Path("requirements.txt").read_text().splitlines(),
)