from setuptools import find_packages, setup

package_name = 'network_model'

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
            'network_model_node = network_model.node_ros:main',
        ],
    },
    maintainer='MP-CARE Project',
    maintainer_email='mpcare@example.com',
    description='Explicit swarm network model: path loss, PRR, ETX, queues, injected loss/latency.',
    license='MIT',
    tests_require=['pytest'],
)
