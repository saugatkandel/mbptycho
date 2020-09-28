from setuptools import setup

setup(
    name='mbptycho',
    version='0.0.1',
    author='Saugat Kandel',
    author_email='saugat.kandel@u.northwestern.edu',
    packages=['mbptycho'],
    #packages=['optimizers', 'tests', 'benchmarks', 'examples'],
    #package_data={'data': ['*.png']},
    scripts=[],
    description='Multi-peak bragg ptychography',
    requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "tensorflow",
        "scikit-image"
    ],
)
