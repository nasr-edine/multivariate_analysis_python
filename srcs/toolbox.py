# Interactive development environments (IDEs) and notebooks
import jupyterlab as jl
import ipywidgets as widgets

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import folium as fm
import plotly as py
import colorama as cr

# Data manipulation libraries
import numpy as np
import pandas as pd

# Statistical modeling libraries
import statsmodels as sm
# import sklearn as sk

def load_dataset(filepath, sep):
  df = pd.read_csv(filepath, sep=sep, low_memory=False)
  return df

def calculate_mean(data):
  mean = data.mean()
  return mean

import os
import sys
from colorama import Fore
    

def get_feature_types(feature_types):
    quantitative_features = [] 
    qualitative_features = [] 
    # Iterate over the feature types 
    for feature, dtype in feature_types.items():     
        # If the data type is float or int, it is quantitative     
        if dtype == float or dtype == int:         
            quantitative_features.append(feature)     
        else:         
            qualitative_features.append(feature) 
    return quantitative_features, qualitative_features
  
def classify_features(features):
    feature_classification = {
        'metadata': [],
        'product_info': [],
        'location_info': [],
        'nutritional_info': []
    }

    for feature in features:
        if '_t' in feature or '_datetime' in feature:
            feature_classification['metadata'].append(feature)
        elif '_name' in feature or '_tags' in feature or '_fr' in feature:
            feature_classification['product_info'].append(feature)
        elif '_places' in feature or '_code_geo' in feature or 'cities' in feature:
            feature_classification['location_info'].append(feature)
        elif '_100g' in feature or '_serving' in feature or '_value' in feature:
            feature_classification['nutritional_info'].append(feature)

    feature_classification_df = pd.DataFrame.from_dict(feature_classification, orient='index').transpose()

    max_length = 20
    feature_classification_df = feature_classification_df.applymap(lambda x: x[:max_length] + "..." if isinstance(x, str) and len(x) > max_length else x)

    return feature_classification_df

def get_fill_rates(df):
    # Calculate the fill rate for each column
    fill_rates = df.count() / df.shape[0]
    
    # Convert the fill rates into a dataframe
    fill_rates_df = pd.DataFrame(fill_rates, columns=['Fill Rate'])
    
    # Sort the dataframe by the Fill Rate column in descending order
    fill_rates_df = fill_rates_df.sort_values(by='Fill Rate', ascending=False)
    
    return fill_rates_df
  
def remove_columns_by_string(df, string):
  # Select the columns that do not contain the string in the column name
  columns_to_remove = df.columns.str.contains(string)
  print(f"The list of features removed based on business approach: {list(df.columns[columns_to_remove])}")

  # Remove the columns from the DataFrame
  df = df.loc[:, ~df.columns.str.contains(string)]
  return df

def pie_chart_features(df, title):
    
    labels = df.columns
    
    # Get the number of non-empty values for each column
    counts = df.count()

    # Calculate the proportion of non-empty values for each column
    proportions = counts / df.shape[0]

    fig, ax = plt.subplots()
    ax.pie(proportions, labels=labels)
    ax.legend(labels, bbox_to_anchor=(1,1))
    # Add a title
    fig.suptitle(title)
    # Show the plot
    plt.show()    


def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    """Display correlation circles, one for each factorial plane"""

    # For each factorial plane
    for d1, d2 in axis_ranks: 
        if d2 < n_comp:

            # Initialise the matplotlib figure
            fig, ax = plt.subplots(figsize=(10,10))

            # Determine the limits of the chart
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # Add arrows
            # If there are more than 30 arrows, we do not display the triangle at the end
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (see the doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # Display variable names
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # Display circle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # Define the limits of the chart
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # Display grid lines
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # Label the axes, with the percentage of variance explained
            plt.xlabel('PC{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('PC{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Correlation Circle (PC{} and PC{})".format(d1+1, d2+1))
            plt.show(block=False)

def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    '''Display a scatter plot on a factorial plane, one for each factorial plane'''

    # For each factorial plane
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # Initialise the matplotlib figure      
            fig = plt.figure(figsize=(7,6))
        
            # Display the points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # Display the labels on the points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # Define the limits of the chart
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # Display grid lines
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # Label the axes, with the percentage of variance explained
            plt.xlabel('PC{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('PC{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection of points (on PC{} and PC{})".format(d1+1, d2+1))
            #plt.show(block=False)
   