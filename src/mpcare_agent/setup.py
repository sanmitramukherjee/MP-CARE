from setuptools import find_packages, setup

package_name = 'mpcare_agent'

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
            'planner_node = mpcare_agent.planner_node:main',
        ],
    },
    maintainer='MP-CARE Project',
    maintainer_email='mpcare@example.com',
    description='MP-CARE core UAV agent: planner, cost engine, link predictor, event trigger.',
    license='MIT',
    tests_require=['pytest'],
)
