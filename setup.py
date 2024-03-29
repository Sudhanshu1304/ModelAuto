from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 6 - Mature',
  'Topic :: Scientific/Engineering :: Artificial Intelligence',
  'Intended Audience :: Education',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3',
  'Programming Language :: Python :: 3.0',
  'Programming Language :: Python :: 3.3',
  'Programming Language :: Python :: 3.4',
  'Programming Language :: Python :: 3.5',
  'Programming Language :: Python :: 3.6',
  'Programming Language :: Python :: 3.7',
  'Programming Language :: Python :: 3.8',
  'Programming Language :: Python :: 3.9'
]
 
setup(
  name='ModelAuto',
  packages=['ModelAuto'],
  version='0.6',
  
  description='Speed up your model making process. This will help you in selecting best features, best Models (SVM,SVR Random Fotest e.t.c and also in Data Preprocessing',
  
  url='https://github.com/Sudhanshu1304/ModelAuto',  
  author='Sudhanshu Pandey',
  author_email='sudhanshu.dpandey@gmail.com',
  license='MIT', 
  classifiers=classifiers,
  download_url='https://github.com/Sudhanshu1304/ModelAuto/archive/v0.6.tar.gz',
  keywords=['Machine Learning','Model','Regression','Classification','Automation','Data Preprocessing','Preprocessing','Feature Selection','Model Selection','SVM'], 

  install_requires=['pandas','numpy','seaborn','matplotlib','statsmodels','sklearn'] 
)