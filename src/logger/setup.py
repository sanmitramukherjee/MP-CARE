from setuptools import find_packages, setup

package_name = 'logger'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'logger_node = logger_pkg.logger_node:main',
        ],
    },
    maintainer='MP-CARE Project',
    maintainer_email='mpcare@example.com',
    description='Swarm metrics logger writing the section 3.4 CSV schema per trial.',
    license='MIT',
    tests_require=['pytest'],
)
