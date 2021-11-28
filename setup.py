from setuptools import setup, find_packages

setup(
    name="spanningtrees",
    version="1.0",
    description="Spanning Tree Algorithms",
    author=["Ran Zmigrod", "Tim Vieira"],
    url="https://github.com/rycolab/spanningtrees",
    install_requires=[
        "numpy",
    ],
    extras_require={"tests": ["pytest", "tqdm"]},
    packages=find_packages(),
)
