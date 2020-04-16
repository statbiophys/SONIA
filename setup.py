from setuptools import setup, find_packages

# def readme():
#     with open('README.md') as f:
#         return f.read()

data_files_to_include = [('', ['README.md', 'LICENSE'])]

setup(name='sonia',
      version='0.0.1',
      description='Infer and compute selection factors of CDR3 sequences',
      long_description='text/markdown',
      url='https://github.com/statbiopys/SONIA',
      author='Zachary Sethna, Giulio Isacchini, Yuval Elhanati',
      author_email='zachary.sethna@gmail.com, giulioisac@gmail.com',
      license='GPLv3',
      classifiers=[
            'Development Status :: 0 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Healthcare Industry',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Scientific/Engineering :: Physics',
            'Topic :: Scientific/Engineering :: Medical Science Apps.',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Natural Language :: English',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.6',
            ],
      packages=find_packages(),
      install_requires=['numpy','tensorflow','matplotlib','olga'],
      package_data = {
            'default_models': [],
            'default_models/human_T_alpha/': ['sonia/default_models/human_T_alpha/*'],
            'default_models/human_T_beta/': ['sonia/default_models/human_T_beta/*'],
            'default_models/mouse_T_beta/': ['sonia/default_models/mouse_T_beta/*'],
            'default_models/human_B_heavy/': ['sonia/default_models/human_B_heavy/*'],
            'default_models/human_B_kappa/': ['sonia/default_models/human_B_kappa/*'],
            'default_models/human_B_lambda/': ['sonia/default_models/human_B_lambda/*'],
            },
      data_files = data_files_to_include,
      include_package_data=True,
      entry_points = {'console_scripts': [
            'sonia-evaluate=sonia.evaluate:main',
            'sonia-generate=sonia.generate:main',
            'sonia-infer=sonia.infer:main'], },
      zip_safe=False)