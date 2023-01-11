import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="moral-arbiter",
    version="1.6",
    author="Stephan Zheng, Alex Trott, Sunil Srinivasa",
    author_email="mhelabd@stanford.edu",
    description="Using Multi-Agent Reinforcement Learning to Simulate Moral Theories",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mhelabd/moral-arbiter",
    packages=setuptools.find_packages(),
    package_data={
        "moral_arbiter": [
            "foundation/scenarios/simple_wood_and_stone/map_txt/*.txt",
            "foundation/components/*.cu",
        ],
    },
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
