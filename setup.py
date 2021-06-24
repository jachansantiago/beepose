from setuptools import find_packages, setup

setup(
    name='beepose',
    packages=find_packages(),
    version='0.1.1',
    description='After rewriting the code of the pose estimation I will create a new one. ',
    author='Ivan Felipe rodriguez',
    license='BSD-3',
    scripts=['scripts/beepose', 'scripts/docker-beepose'],
    entry_points={
        'console_scripts': [
            'train_stages_aug = beepose.train.train_stages_aug:main',
            'process_folder_full_video = beepose.inference.process_folder_full_video:main',
            'get_inference_model = beepose.train.get_inference_model:main'
        ],
    }
)
