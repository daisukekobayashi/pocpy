import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pocpy",
    version="0.1.0",
    license="MIT",
    author="Daisuke Kobayashi",
    author_email="daisuke@daisukekobayashi.com",
    description="Phase Only Correlation in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/daisukekobayashi/pocpy",
    packages=setuptools.find_packages(),
    kewords="registration phase-only-correlation",
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=2.7,!=3.0.*,!=3.1.*,!=3.2.*",
    install_requires=["six", "numpy", "scipy", "opencv-python"],
)
