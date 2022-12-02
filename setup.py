from setuptools import find_packages
from setuptools import setup

with open("requirements_prod.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='deep_draw',
      version="0.0.1",
      description="Deep Draw Project",
      license="MIT",
      author="Sythak",
      author_email="contact@lewagon.org",
      url="https://github.com/Sythak/deepdraw",
      install_requires=requirements,
      packages=find_packages(),
      )
