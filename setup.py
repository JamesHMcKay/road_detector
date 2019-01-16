from setuptools import setup, find_packages

with open('README.md') as f:
    readme_text = f.read()

with open('LICENSE') as f:
    license_text = f.read()

setup(
    name='road_detector',
    version='0.1.0',
    description='Detect road networks from satellite images',
    long_description=readme_text,
    author='James McKay',
    author_email='jhmckay93@gmail.com',
    url='https://github.com/JamesHMcKay/road_detector',
    license=license_text,
    packages=find_packages(exclude=('tests', 'data'))
)
