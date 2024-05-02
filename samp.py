# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import requests

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV

class cancer_cell_data(object):
    """This class is used to create an object which will hold the 
    cancer cell data in a pandas data frame.  It will overload some
    features of the dataframe just for convience.  The constructor
    will fetch the data from the url.
    
    self.df - Pandas dataframe."""
    
    col_names = ['ID','thickness', 'unif. size', 'unif. shape',\
                 'adhesion','epithelial size', 'nuclei',\
                 'chromatin', 'nucleoli', 'mitosis',
                 'class']
    
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'
    url+= 'breast-cancer-wisconsin/breast-cancer-wisconsin.data'
    
    def __init__(self):
        """Reads in the data from the url and assigns the column names
        to be names in headers. Also sets the index to be the ID. """
        self.df = pd.read_csv(self.url, names=self.col_names,)
        self.df.set_index('ID', inplace=True)
        self.means = None
        self.stds  = None
        
    def head(self):
        """Returns the first five lines of the data frame."""
        return self.df.head()
    
    def rows(self):
        """Returns the number of rows in the pandas data frame."""
        return self.df.shape[0]
        
    def cols(self):
        """Returns the number of columns in the pandas data frame."""
        return self.df.shape[1]
    
    def describe(self):
        """Returns the descriptive statistics of the dataset without 
        suppressing the NANS if there are any."""
        return self.df.describe(include='all')
    
    def clean(self):
        """Replace the ? with nans, then print the number of nans
        to show there arent many, and delete them."""
        
        # replace the ? with nans
        self.df.replace('?', np.nan, inplace=True)
        
        # convert the nuclei values to numbers since they are
        # actually stored as strings
        self.df['nuclei'] = pd.to_numeric(self.df['nuclei'])
        
        # print the number of characters in each
        self.df.dropna(inplace=True)
        
        # drop the ID's
        self.df.reset_index(inplace=True)
        self.df.drop('ID',axis=1,inplace=True)
        
        # set benign = 0, malignant = 1
        for i in range(0,self.df.shape[0]):
            if(self.df.loc[i,'class'] == 2):
                self.df.loc[i,'class'] = 0
            else:
                self.df.loc[i,'class'] = 1
                
                
    def scale(self):
        """Normalize the data so that the columns have zero mean
        and unit variance. Stores the column means and stds so
        that you can rescale back."""
        self.means = self.df.mean()
        self.stds  = self.df.std()
        self.df = (self.df - self.means ) / self.stds
        
        
   
class data_visualizer(object):
    ''' This class is just a container for different visualizations
    of the data stored in cancer_cell_data object.'''
    
    def __init__(self):
        """Empty constructor."""
    
    def dist_of_cells(self, data):
        """Makes a histogram of the number of cancer cells and
        normal cells."""
        plt.figure(figsize=(6,4))
        data.df['class'].hist()
        plt.ylabel('Number of Targets',fontsize=12)
        plt.xlabel('Class of cell',fontsize=12)
        plt.title('Histogram of number cells that are cancerous',
                  fontsize=12)
        
    def corr(self, data):
        """Makes a heatmap of the correlation matrix."""
        sns.heatmap(data.df.corr())
        
    def dist_of_features(self, data):
        """Plots the histograms and 
        kernal destinsity estimator of the different features."""
      
        sns.set(style="white", palette="muted", color_codes=True)
       
        # Set up the matplotlib figure
        f, axes = plt.subplots(3,3, figsize=(7, 7), sharex=True)
        sns.despine(left=True)
        
        # get the column names
        col_names = data.df.columns.tolist()
        
        # loop through the features and map them to a place in the 
        # figure
        for i in range(0,9):
            row = i / 3
            col = i % 3
            sns.distplot(data.df[col_names[i]],
                         kde=True, color="b", ax=axes[row, col])
        # drop the y-axis label
        plt.setp(axes, yticks=[])
        plt.tight_layout()
        
    def feature_violin(self, data):
        """Produces violin plot of the features with respect to 
        the cell class."""
        for i, col_name in enumerate(data.df.columns[:-1]):
            plt.figure(i)
            sns.factorplot(x="class",col=col_name,
                           data=data.df,kind="violin")
     
        
data = cancer_cell_data()
viz = data_visualizer()
viz.dist_of_cells(data)

viz.corr(data)