import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="scripts",
    version="0.0",
    author="Rocco Meli",
    license="MIT",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",  # GitHub-flavored Markdown
    packages=setuptools.find_packages(),
    install_requires=[],
    python_requires=">=3.6",
)
