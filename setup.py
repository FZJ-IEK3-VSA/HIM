from setuptools import setup, find_packages

setup(
    name='HIM',
    version='1.0',
    description='Evaluation of hydrogen infrastructure via geospatial datatools.',
 
    url='http://www.fz-juelich.de/iek/iek-3/DE/Home/home_node.html',
    author='Markus Reuss',
    author_email='m.reuss@fz-juelich.de',    
    license='',
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        'CoolProp',
        'geopandas',
        'jupyter',
        'networkx<2.0',
        'openpyxl',
        'Pyomo',
        'xlrd',
        'XlsxWriter',
	'descartes',
    ]
)