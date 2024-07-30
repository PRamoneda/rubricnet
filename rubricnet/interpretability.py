import argparse
import math
import os
import pdb
from itertools import chain, combinations
from statistics import mean, stdev
from time import sleep

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import pandas as pd
import six
import sklearn
import torch

from mpmath import norm
from scipy.stats import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
import wandb
from optuna_bayesian_optimization import music21_features

from rubricnet import RubricnetSklearn, _prediction2label

import plotly.graph_objects as go
import pandas as pd

all_columns = [
  'duration',
  'unextended_duration',
  'pitch_entropy',
  'pitch_set_entropy',
  'pitch_class_set_entropy',
  'pitch_set_entropy_rate',
  'pitch_set_lz',
  'pitch_class_set_entropy_rate',
  'rh_pitch_set_entropy',
  'rh_pitch_class_set_entropy',
  'rh_pitch_set_entropy_rate',
  'rh_pitch_set_lz',
  'rh_pitch_class_set_entropy_rate',
  'lh_pitch_set_entropy',
  'lh_pitch_class_set_entropy',
  'lh_pitch_set_entropy_rate',
  'lh_pitch_set_lz',
  'lh_pitch_class_set_entropy_rate',
  'pitch_class_entropy',
  'new_rh_stamina',
  'new_lh_stamina',
  'new_joint_stamina',
  'rh_pitch_range',
  'lh_pitch_range',
  'pitch_range',
  'lh_average_pitch',
  'rh_average_pitch',
  'average_pitch',
  'lh_average_ioi_seconds',
  'rh_average_ioi_seconds',
  'joint_average_ioi_seconds',
  'rh_displacement_rate',
  'lh_displacement_rate',
  'displacement_rate'
]


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import utils

from collections import namedtuple, defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gamma, norm, expon, poisson
import matplotlib.pyplot as plt
import math

def get_acc1_macro(y_true, y_pred):
    acc_plusless_1_each_class = []
    for true_class in set(y_true):
        matches = [1 if pp in [tt - 1, tt, tt + 1] else 0 for tt, pp in zip(y_true, y_pred) if tt == true_class]
        acc_plusless_1_each_class.append(sum(matches) / len(matches))
    return mean(acc_plusless_1_each_class)


def get_mse_macro(y_true, y_pred):
    mse_each_class = []
    for true_class in set(y_true):
        tt, pp = zip(*[[tt, pp] for tt, pp in zip(y_true, y_pred) if tt == true_class])
        mse_each_class.append(mean_squared_error(y_true=tt, y_pred=pp))
    return mean(mse_each_class)





minimal_columns_chiu = [
 'rh_pitch_entropy', # if the distribution of the pitches close to uniform
 'lh_pitch_entropy',
 'rh_pitch_range', # trivial
 'lh_pitch_range',
 'rh_average_pitch', # trivial
 'lh_average_pitch',
 'rh_average_ioi_seconds', # bpm is not very defined
 'lh_average_ioi_seconds',
 'rh_displacement_rate', #
 'lh_displacement_rate'
]

# 	38.12(5.05)	70.26(1.32)	81.18(2.85)	1.63(0.38)
minimal_columns_basic = [
     'rh_pitch_entropy',
     'lh_pitch_entropy',
     'rh_pitch_set_lz',
     'lh_pitch_set_lz', # yes
]

minimal_columns_total = [
    'rh_pitch_set_lz',
    'lh_pitch_set_lz',
    'rh_pitch_range',
    'lh_pitch_range',
    'lh_average_pitch',
    'rh_average_pitch',
    'lh_average_ioi_seconds',
    'rh_average_ioi_seconds',
    'rh_displacement_rate',
    'lh_displacement_rate',
    'lh_pitch_entropy',
    'rh_pitch_entropy'
]



def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(1, len(ss)+1)))
    #return ['pitch_entropy', 'rh_pitch_set_lz', 'rh_pitch_range', 'lh_pitch_range', 'pitch_range', 'average_pitch', 'displacement_rate']


def plot_descriptor_scores_vs_values(scores, values, columns):
    """
    Plots line graphs for multiple descriptors with each descriptor's values on the x-axis
    and the corresponding scores on the y-axis. Each line represents one descriptor and will
    have a unique color.

    Parameters:
    - scores: List of lists, where each inner list contains scores for a descriptor (12 x number of samples).
    - values: List of lists, where each inner list contains values for a descriptor (12 x number of samples).
    - columns: List of strings, names of the descriptors.
    """
    values = values.T.tolist()
    # Check if the number of descriptors in 'scores' and 'values' matches the number of names in 'columns'
    if len(scores) != len(values) or len(scores) != len(columns):
        print("Error: The number of descriptors, scores, and column names must match.")
        return

    # Generate a list of colors for the plots
    colors = [
        "#FF0000", "#FF7F00", "#FFFF00", "#7FFF00",
        "#00FF00", "#00FF7F", "#00FFFF", "#007FFF",
        "#0000FF", "#7F00FF", "#FF00FF", "#FF007F"
    ]

    plt.figure(figsize=(10, 6))

    # Plot each descriptor's scores against its values
    for i in range(len(scores)):
        plt.plot(values[i], scores[i], label=columns[i], color=colors[i], marker='o', linestyle='None')

    plt.title('Descriptor Scores vs. Descriptor Values')
    plt.xlabel('Descriptor Values')
    plt.ylabel('Descriptor Scores')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def plot_regression_outputs_by_grade_green(grades, regression_values):
    """
    Plots a box plot of regression outputs by grade with all boxes colored in green.

    Parameters:
    - grades: A list of integer grades.
    - regression_values: A list of regression output values corresponding to each grade.
    """
    # Grouping regression values by their corresponding grades
    grouped_data = defaultdict(list)
    for grade, value in zip(grades, regression_values):
        grouped_data[grade].append(value)

    # Sorting the dictionary by grade to ensure the box plots are ordered
    sorted_keys = sorted(grouped_data.keys())
    sorted_values = [grouped_data[key] for key in sorted_keys]

    # Set up the figure and axis for the box plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Creating the box plot with green color
    boxprops = dict(facecolor='green', color='green')
    whiskerprops = dict(color='green')
    capprops = dict(color='green')
    medianprops = dict(color='white')
    flierprops = dict(marker='o', color='green', alpha=0.5)

    # Plotting the boxes
    ax.boxplot(sorted_values, patch_artist=True, labels=sorted_keys,
               boxprops=boxprops, whiskerprops=whiskerprops,
               capprops=capprops, medianprops=medianprops,
               flierprops=flierprops)

    # Setting the labels and title
    ax.set_xlabel('Grades')
    ax.set_ylabel('Regression Output Values')
    plt.title('Box Plot of Regression Outputs by Grade')

    # Show plot
    plt.show()


def render_table_with_darker_header_and_final(data, col_width=2.5, row_height=0.4, font_size=16,
                                              header_color='#D8E2DC', #'#0047ab', # Using a darker blue color
                                              row_colors=['#f1f1f2', 'w'], edge_color='w',
                                              bbox=[0, 0, 1, 1], header_columns=0,
                                              ax=None, **kwargs):
    if ax is None:
        size = (np.array([len(data[0]), len(data)]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
        ax.set_title('Local explanations RubricNet', fontsize=12, weight='bold')
    table_data = [['Descriptor Name', 'Hand', 'Descriptor Value', 'Score', 'Grade Divergence', 'Accumulative Score']] + data
    mpl_table = ax.table(cellText=table_data, bbox=bbox, colLabels=None, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[0] == len(data):  # Apply darker blue style to both header and final score row
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return fig, ax


# Function to create table for given descriptor scores and save to specified path

def create_and_save_table(descriptor_scores, split, name, columns, descriptors, mean_score_level, save_dir="local_explainability"):
    # Calculate accumulative score
    accumulative_score = 0
    # Generate table data
    table_data = []
    for idx, score in enumerate(descriptor_scores):
        feature_name = columns[idx].replace('rh_', '').replace('lh_', '').replace('_', ' ').title()
        hand = 'Right' if 'rh' in columns[idx] else 'Left'
        score_with_sign = f"+{round(score, 2)}" if score > 0 else round(score, 2)
        accumulative_score += score
        table_data.append([feature_name, hand, round(descriptors[idx], 2), score_with_sign, round(score - mean_score_level[idx], 2), round(accumulative_score, 2)])
        # Decrease for the next item
    # Add the final score row
    table_data.append(['Final Score', '—', '—', '—', '—', round(sum(descriptor_scores), 2)])
    # Create directories
    save_path = os.path.join(save_dir, str(split))
    os.makedirs(save_path, exist_ok=True)
    # Render and save the table
    fig, ax = render_table_with_darker_header_and_final(table_data, col_width=2.0, row_height=0.5, font_size=8)
    fig.savefig(os.path.join(save_path, f"{name}_table.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)  # Close the plot to free memory


def plot_explorer(ground_truth_grades, prediction_grades, ids, descriptors, columns, split):
    index = utils.load_json("index.json")
    new_columns = [
        'Pitch Entropy (R)', 'Pitch Entropy (L)',
        'Pitch Range (R)', 'Pitch Range (L)',
        'Average Pitch (R)', 'Average Pitch (L)',
        'Average IOI (R)', 'Average IOI (L)',
        'Displacement Rate (R)', 'Displacement Rate (L)',
        'Pitch Set LZ (R)', 'Pitch Set LZ (L)'
    ]

    nsamples = len(descriptors)  # Number of samples

    # Convert descriptor list into a DataFrame
    descriptors_df = pd.DataFrame(descriptors, columns=columns)
    # change columns to new_columns
    descriptors_df.columns = new_columns

    # Convert your initial data into a DataFrame
    data = pd.DataFrame({'ID': ids, 'Ground Truth': ground_truth_grades, 'Prediction': prediction_grades})

    # Concatenate data with descriptors along the columns
    df = pd.concat([data, descriptors_df], axis=1)

    # Define a color palette
    color_palette = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
                     'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)']
    color_map = {feature: color_palette[i % len(color_palette)] for i, feature in enumerate(new_columns)}

    # Create a box plot for each feature
    fig = go.Figure()
     # Start with the first feature visible
    # visible = True
    for feature in new_columns:  # Assuming 'new_columns' is your list of features
        for grade in sorted(df['Ground Truth'].unique()):
            df_filtered = df[df['Ground Truth'] == grade]
            # Add box trace for each grade of the current feature
            fig.add_trace(go.Box(
                y=df_filtered[feature],
                x=[f'Grade {grade}' for _ in range(len(df_filtered))],
                name=feature,  # Used for legend entries
                marker_color=color_map[feature],  # Assign color
                boxpoints='all', jitter=0.5, pointpos=-1.8,
                hoverinfo='text',
                text=df_filtered.apply(lambda row: f'ID: {row["ID"]}'
                                                   f'<br>Prediction: {row["Prediction"]}'
                                                   f'<br>Composer: {index[row["ID"]]["composer"]}'
                                                   f'<br>{index[row["ID"]]["work_name"]}', axis=1),
                legendgroup=feature,  # Group by feature for toggling
                showlegend=True if grade == min(df['Ground Truth'].unique()) else False
            ))
            fig.update_traces(visible="legendonly")
        # visible = False
    # Customize layout
    fig.update_layout(
        title='Box Plot of Features by Ground Truth Grade',
        xaxis=dict(title='Feature - Grade', type='category'),
        yaxis=dict(title='Value'),
        boxmode='group',  # Group boxes of the same features across different grades
        legend_title='Feature',
        legend=dict(tracegroupgap=0),
    )
    # Define the path to save the plot HTML
    save_path = f'local_explainability/explorer_score_{split}.html'
    visible = "legendonly"

    # Save the plot
    fig.write_html(save_path)
    sleep(5)
    # exit()

def generate_local_explainability(descriptor_scores, descriptors, split, ids_test, columns, ground_truth_grades, prediction_grades):
    # Loop through each sample and its descriptor scores to generate and save tables
    new_descriptor_scores = []
    for idx in range(len(descriptor_scores[0])):
        new_descriptor_scores.append([dd[idx] for dd in descriptor_scores])
    # Calculate mean score level for each ground_truth_grades
    mean_score_level = {grade - 1: [
        mean([descriptor_scores[jdx][idx] for idx in [idx for idx in range(len(descriptor_scores[0])) if grade == ground_truth_grades[idx]]])
        for jdx in range(12)
        ] for grade in set(ground_truth_grades)
    }

    for name, scores, dd, gt in zip(ids_test, new_descriptor_scores, descriptors.values.tolist(), ground_truth_grades):
        create_and_save_table(scores, split, name, columns, dd, mean_score_level=mean_score_level[gt-1], save_dir="local_explainability")
    plot_explorer(ground_truth_grades, prediction_grades, ids_test, new_descriptor_scores, columns, split)


def make_boundaries_from_roots(roots, increasing: bool = True):
    """make_boundaries_from_roots.
    Based on roots of the sigmoids of the output layer, determines the boundaries of each grade.
    The term root is used improperly here because it designates x where s(x) = 0.5, so that
    the new grade is always active.

    Args:
        roots: list of roots for each sigmoid of the final output stage.
        increasing (bool): Flag regarding the order of the scores for each grade.
    Returns:
        boundaries: [(lower, upper), ...] for each grade.
    """
    boundaries = {}
    if increasing:
        if roots[-1] < roots[-2]:
            # If final grade is missing, its root value breaks the order
            roots.pop()
        # boundaries are shifted to always be in [0,12]
        boundaries[0]=(0, (roots[1].item()+12)/2)
        for i in range(1,len(roots)-1):
             boundaries[i]=((roots[i].item()+12)/2, (roots[i+1].item()+12)/2)
        # Last one always go up to 12
        boundaries[len(boundaries.keys())]=((roots[-1].item()+12)/2, 12)
    else:
        if roots[-1] > roots[-2]:
            # If final grade is missing, its root value breaks the order
            roots.pop()
        # take opposite of roots to shift boundaries back to [0,12]
        roots = [-r for r in roots]
        boundaries = make_boundaries_from_roots(roots, True)
    return boundaries



def calculate_class_boundaries(grades, clf,
        min_val: int = -12, max_val: int = 12):
    """
    Calculates the lower and upper boundaries for each grade based on the current classifier.

    Parameters:
    - grades: A list of integer grades.
    - clf: The classifier to use.
    - min_val: lowest value possible for regression values. Default is -12.
    - max_val: highest value possible for regression values. Default is 12.

    Returns:
    A dictionary with grades as keys and (lower_boundary, upper_boundary) as values.
    """
    # Get root value for each grade
    roots = []
    biases = clf.model.linear1.final_layer.bias
    weights = clf.model.linear1.final_layer.weight
    for g in grades:
        roots.append((0-biases[g])/weights[g])
    increasing = True
    # Check if roots are ordered increasingly.
    # we skip first and last element that can be wrongly ordered.
    for i in range(2, len(roots)-1):
        if roots[i] < roots[i-1]:
            increasing = False
            break
    class_boundaries = make_boundaries_from_roots(roots, increasing)
    class_boundaries = {k+1: v for k, v in class_boundaries.items()}
    return class_boundaries


def display_table_with_intervals(data, split, save_path="local_explainability"):
    # Define table styling parameters
    header_color = '#D8E2DC'  # Dark blue color for the first column
    font_size = 10
    row_colors = ['#f1f1f2', 'w']
    edge_color = 'w'

    # Prepare table header and data
    table_header = ['GRADE', 'BOUNDARIES']
    sorted_data = sorted(data.items(), key=lambda item: int(item[0]))
    table_rows = [[str(key), f"{min_val:.2f} to {max_val:.2f}"] for key, (min_val, max_val) in sorted_data]
    table_data = [table_header] + table_rows
    table_data = np.array(table_data).T

    # Determine figure size
    num_rows, num_cols = 2, len(table_data[0])
    fig_width, fig_height = 15, 1
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    # ax.set_title('Global Boundaries RubricNet', fontsize=12, weight='bold')

    # Create the table
    mpl_table = ax.table(cellText=table_data, loc='center', cellLoc='center', edges='closed')
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    # Apply styling to cells
    for k, cell in six.iteritems(mpl_table._cells):
        if k[1] == 0:  # First column
            cell.set_text_props(color='black', weight='bold')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
        cell.set_edgecolor(edge_color)

    plt.show()
    fig.savefig(os.path.join(save_path, f"boundaries_{split}.png"), bbox_inches='tight', dpi=300)
    plt.close(fig)  # Close the plot to free memory


def plot_boundaries(final_boundaries):
    columns = np.arange(1, 10)  # Grade numbers
    splits = ["1", "2", "3", "4", "5"]
    plt.rcParams.update({'font.size':24})

    boundaries = np.full((5, 18), np.nan)
    for s in splits:
        key = int(s)-1
        for k, v in enumerate(final_boundaries[key].values()):
            low, high = v
            boundaries[key][k] = low
            boundaries[key][k+9] = high

    # Adjusting the visualization with larger text sizes and labels

    fig, ax = plt.subplots(figsize=(14, 8))  # Increased figure size for better visibility

    # Redefine initial settings with larger text sizes
    y_base = np.linspace(0,0.8,num=len(splits))[::-1]
    cmap = get_cmap('Greens')
    colors = cmap(np.linspace(0.3, 0.9, len(columns)))  # Adjusted color range for better visibility
    #colors = ['#a74e0f', '#a96222', '#ad7332', '#b4823f', '#bc914a', '#c89e53', '#d7aa5a', '#eab660', '#ffc064'][::-1]

    # Apply larger text sizes
    for i, split in enumerate(splits):
        for j in range(9):  # Iterate through each grade
            start_point = boundaries[i][j]
            end_point = boundaries[i][j + 9]
            width = end_point - start_point  # Width of the bar is the range of boundary values
            print(start_point, end_point, width)

            if not np.isnan(start_point) and not np.isnan(end_point):
                ax.barh(y_base[i], width, left=start_point, height=0.12, color=colors[j], edgecolor='black')
                mid_point = start_point + (width / 2)
                if j <= 3:
                    c = 'black'
                else:
                    c = 'white'
                ax.text(mid_point, y_base[i], str(j + 1), ha='center', va='center', color=c)  # Increased font size

    # Adjusting the plot aesthetics with larger labels
    ax.set_yticks(y_base)
    ax.set_yticklabels(splits
                      )  # Increased font size for y-tick labels
    #ax.set_xlabel(r'$S_\text{agg}$', fontsize=20)  # Increased font size for the x-axis label
    #ax.set_ylabel('Split', fontsize=20)  # Increased font size for the x-axis label


    # Hide the right and top spines
    ax.spines[['right', 'top']].set_visible(False)
    #ax.set_title('Sequential Placement of Grade Boundaries for Each Split', fontsize=16)  # Increased title font size
    plt.tight_layout()
    plt.savefig("boundaries_bars.pdf")
    plt.show()


def get_contributions(pred_test, pred_scores, s:int = 0,features=None):
    contributions = [{k: np.array([]) for k in features} for g in range(9)]
    sum_score_by_grades = [np.array([]) for _ in range(9)]
    normalized_contrib = [{k: np.array([]) for k in features} for g in range(9)]
    for i in range(len(pred_test)):
        if s in [1, 2, 4]:
            scores = [pred_scores[f][i] + 1 for f in range(len(features))]
        else:
            scores = [1 - pred_scores[f][i] for f in range(len(features))]
        g = pred_test[i]
        for f, name in enumerate(features):
            contributions[g][name] = np.append(contributions[g][name],scores[f])
        sum_score = np.sum(np.abs(scores))
        sum_score_by_grades[g] = np.append(sum_score_by_grades[g],sum_score)
        # get normalized contributions
        for f, name in enumerate(features):
            norm = scores[f] / 24
            normalized_contrib[g][name] = np.append(normalized_contrib[g][name], norm)
    # average contributions
    avg_norm_contrib = [{k: 0.0 for k in features} for _ in range(9)]
    diff_norm_contrib = [{k: 0.0 for k in features} for _ in range(9)]
    for g in range(9):
        for f, name in enumerate(features):
            avg_norm_contrib[g][name] = normalized_contrib[g][name].mean()
            if g == 0:
                pass
            else:
                diff_norm_contrib[g][name] = (avg_norm_contrib[g][name] - avg_norm_contrib[0][name])
    return avg_norm_contrib, diff_norm_contrib

def plot_contrib(contrib, split:int=0,feat=None):
    cmap = get_cmap('Greens')
    COLORS = cmap(np.linspace(0.3, 0.9, 6))  # Adjusted color range for better visibility
    STROKES = ['', '/']

    y_labels = ["P. Entropy (R)", "P. Entropy (L)",
                "P. Range (R)", "P. Range (L)",
               "Avg P. (R)", "Avg P. (L)",
               "Avg IOI (R)", "Avg IOI (L)",
               "Disp. Rate (R)", "Disp. Rate (L)",
               "P. Set LZ (R)", "P. Set LZ (L)"]
    fig, ax = plt.subplots(figsize=(6,4))
    # Plot stacked bars
    for g in range(9):
        step = 12
        bottom = 0
        for f,name in enumerate(feat[::-1]):
            val = contrib[g][name]
            ax.bar(g, val, 0.5, label=name, bottom=bottom, color=COLORS[f//2%6], hatch=STROKES[f%2],edgecolor='black')
            bottom += 1/step
    # draw horizontal lines
    ax.hlines(np.arange(0,12,1)/step,-1,9, color='black',linewidth=0.5)
    plt.yticks(np.arange(0,12,1)/step,y_labels[::-1])
    # fix legend
    handles, labels = ax.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)
    # show zero line
    #ax.legend(handle_list, label_list,ncol=2,loc='upper left',)
            #bbox_to_anchor=(0,0.9))
    plt.xticks(np.arange(0, 9, 1.0),np.arange(1, 10, 1))
    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xlabel("Grade")
    ax.set_ylabel("")
    plt.ylim(0,12/step)
    plt.xlim(-0.5,8.5)
    plt.tight_layout()
    #plt.savefig(f'figs/contribution_{s}.pdf',transparent=False)
    return ax, fig

def plot_avg(contribs, features_selected=minimal_columns_total):
    contrib = [{k: np.array([]) for k in features_selected} for g in range(9)]
    for g in range(9):
        for f, name in enumerate(features_selected):
            avg = 0
            count = 0
            for s in range(5):
                val = contribs[s][g][name]
                if not(np.isnan(val)):
                    avg += val
                    count += 1
            avg = avg / count
            contrib[g][name] = avg
    ax, fig =plot_contrib(contrib, 'all', feat=features_selected)
    return ax, fig

def main(columns, alias_experiment, dataset="cipi"):
    if dataset == "cipi":
        data = utils.load_json("cipi_splits.json")
        if FEATURES == "jsymbolic":
            features = {v["id"]: v for v in utils.load_json("../features/jSymbolic-CIPI.json")}
        elif FEATURES == "music21":
            features = {v["id"]: v for v in utils.load_json("../features/music21-CIPI.json")}
        elif FEATURES == "all":
            features = {v["id"]: v for v in utils.load_json("../features/all-CIPI.json")}
        else:
            features = {v["id"]: v for v in utils.load_json("../features/current_difficulties.json")}

    final_acc9, final_acc3, final_acc1, final_mse, final_best_selection = [], [], [], [], []
    final_boundaries = []
    contribs = []

    for split in range(5):
        ids_train = data[str(split)]['train'].keys()
        ids_val = data[str(split)]['val'].keys()
        ids_test = data[str(split)]['test'].keys()
        y_train = pd.Series(data[str(split)]['train'].values())
        y_val = pd.Series(data[str(split)]['val'].values())
        y_test = pd.Series(data[str(split)]['test'].values())
        best_acc_val = 0
        # for features_selected in tqdm(all_subsets(columns)):
        features_selected = list(columns)
        X_train = pd.DataFrame({ft: [features[f][ft] for f in ids_train] for ft in features_selected})
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_val = pd.DataFrame({ft: [features[f][ft] for f in ids_val] for ft in features_selected})
        X_test = pd.DataFrame({ft: [features[f][ft] for f in ids_test] for ft in features_selected})

        clf = RubricnetSklearn(input_dim=len(features_selected), num_classes=9, split=split,
                                          args=args)
        #clf.fit(scaler.transform(X_train), y_train, scaler.transform(X_val), y_val, scaler.transform(X_test), y_test)

        # load from checkpoint
        clf.load_model(f"../checkpoints/{alias_experiment}/split_{split}.ckpt")
        pred_val = clf.predict(scaler.transform(X_val))
        X_test_scaled = scaler.transform(X_test)
        pred_test = clf.predict(X_test_scaled)
        regression_test = clf.predict_regression(X_test_scaled)
        descriptor_scores = clf.predict_descriptor_scores(X_test_scaled)

        if split in [0, 3]:
             regression_test *= -1
             regression_test = (regression_test + 12) / 12
             descriptor_scores = [((((dd* -1)+1)/2)).tolist() for dd in descriptor_scores]
        else:
             regression_test = (regression_test + 12) / 12
             descriptor_scores = [((dd+1)/2).tolist() for dd in descriptor_scores]


        plot_regression_outputs_by_grade_green((pred_test+1).tolist(), regression_test.tolist())
        #boundaries = calculate_class_boundaries((pred_test+1).tolist(), regression_test.tolist())
        #display_table_with_intervals(boundaries, split)
        #plot_descriptor_scores_vs_values(descriptor_scores, X_test_scaled, columns)
        #generate_local_explainability(descriptor_scores,X_test, split, ids_test, columns, (y_test+1).tolist(), (pred_test+1).tolist())
        boundaries = calculate_class_boundaries(torch.arange(0,9,1), clf)
        final_boundaries.append(boundaries)
        display_table_with_intervals(boundaries, split)
        plot_descriptor_scores_vs_values(descriptor_scores, X_test_scaled, columns)
        # generate_local_explainability(descriptor_scores,X_test, split, ids_test, columns, (y_test+1).tolist(), (pred_test+1).tolist())
        _, d_contrib = get_contributions(pred_test, descriptor_scores, split, minimal_columns_total)
        contribs.append(d_contrib)

        acc9 = balanced_accuracy_score(y_true=y_test, y_pred=pred_test)
        nine2three = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        acc3 = balanced_accuracy_score(y_true=[nine2three[x] for x in y_test],
                                       y_pred=[nine2three[x] for x in pred_test])
        acc1 = get_acc1_macro(y_true=y_test, y_pred=pred_test)
        mse = get_mse_macro(y_true=y_test, y_pred=pred_test)

        final_acc9.append(acc9)
        final_acc3.append(acc3)
        final_acc1.append(acc1)
        final_mse.append(mse)
        print("best result", best_acc_val, final_acc9)

    plot_boundaries(final_boundaries)
    ax, fig = plot_avg(contribs)
    plt.show()
    fig.savefig("contributions.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"Experiment \t"
            f"{mean(final_acc9) * 100:0.2f}({stdev(final_acc9) * 100:0.2f})\t"
            f"{mean(final_acc3) * 100:0.2f}({stdev(final_acc3) * 100:0.2f})\t"
            f"{mean(final_acc1) * 100:0.2f}({stdev(final_acc1) * 100:0.2f})\t"
            f"{mean(final_mse):0.2f}({stdev(final_mse):0.2f})\t")


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Train a logistic regression model with custom settings.")

    # Add arguments
    parser.add_argument("--alias_experiment", type=str, default="rubricnet_cameraready", help="Alias name for the experiment.")
    parser.add_argument("--set_features", type=str, default="basic", help="Parser Feature set.")
    parser.add_argument("--batch_size", type=int, default=40, help="Batch size for training. Default is 1.")
    parser.add_argument("--patience", type=int, default=400, help="Patience for early stopping. Default is 400.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for the optimizer. Default is 0.1.")
    parser.add_argument("--hidden_size", type=int, default=1, help="Learning rate for the optimizer. Default is 0.1.")
    parser.add_argument("--num_layers", type=int, default=1, help="Learning rate for the optimizer. Default is 0.1.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Learning rate for the optimizer. Default is 0.1.")
    parser.add_argument("--decay_lr", type=float, default=0.9, help="Learning rate for the optimizer. Default is 0.1.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Learning rate for the optimizer. Default is 0.1.")

    # Parse the arguments
    args = parser.parse_args()

    return args


FEATURES = "basic"
if __name__ == '__main__':
    args = parse_args()
    columns = minimal_columns_total
    main(columns=columns, dataset='cipi', alias_experiment=args.alias_experiment)