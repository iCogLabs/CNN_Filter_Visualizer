from distutils.core import setup
import setuptools

with open("../../README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "Visualizing-CNNS",
    version = "1.1",
    py_modules = ["CNNFilterVisualizer",],
    packages=setuptools.find_packages(),
    # Metadata

    author = "Henok Tilaye",
    author_email = "henoktilaye7@gmail.com",
    description = "A modle for visualizing filter activations from gradient values with guided backpropagation",
    long_description_content_type = "text/markdown",
    long_description = long_description,
    license = "Public_domain",
    keywords = "cnn layer activation visualizations",
    url="https://github.com/Hen0k/CNN_Filter_Visualizer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)