from setuptools import setup, find_packages


data_files_to_include = [('', ['README.md', 'LICENSE'])]

setup(name='sonia',
      version='0.0.41',
      description='Infer and compute selection factors of CDR3 sequences',
      long_description='SONIA is a python 3.6/2.7 software developed to infer selection pressures on features of amino acid CDR3 sequences. The inference is based on maximizing the likelihood of observing a selected data sample given a representative pre-selected sample. This method was first used in Elhanati et al (2014) to study thymic selection. For this purpose, the pre-selected sample can be generated internally using the OLGA software package, but SONIA allows it also to be supplied externally, in the same way the data sample is provided. SONIA takes as input TCR CDR3 amino acid sequences, with or without per sequence lists of possible V and J genes suspected to be used in the recombination process for this sequence. Its output is selection factors for each amino acid ,(relative) position , CDR3 length combinations, and also for each V and J gene choice. These selection factors can be used to calculate sequence level selection factors which indicate how more or less represented this sequence would be in the selected pool as compared to the the pre-selected pool. These in turn could be used to calculate the probability to observe any sequence after selection and sample from the selected repertoire. ',
      url='https://github.com/statbiophys/SONIA',
      author='Zachary Sethna, Giulio Isacchini, Yuval Elhanati',
      author_email='zachary.sethna@gmail.com, giulioisac@gmail.com',
      license='GPLv3',
      classifiers=[
            'Development Status :: 3 - Alpha',
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
      install_requires=['numpy','tensorflow>=2.1.0','matplotlib','olga>=1.2.3','tqdm'],
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
