from setuptools import find_packages, setup

package_name = 'aodv_routing'

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
    maintainer='MP-CARE Project',
    maintainer_email='mpcare@example.com',
    description='AODV (RFC 3561) baseline routing — implemented in Phase 2.',
    license='MIT',
    tests_require=['pytest'],
)
