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
    boundaries = {}
    if increasing:
        if roots[-1] < roots[-2]:
            roots.pop()
        boundaries[0]=(0, (roots[1].item()+12)/2)
        for i in range(1,len(roots)-1):
             boundaries[i]=((roots[i].item()+12)/2, (roots[i+1].item()+12)/2)
        boundaries[8]=((roots[-1].item()+12)/2, 12)
    else:
        if roots[-1] > roots[-2]:
            roots.pop()
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

        # if split in [2, 3, 4]:
        #     regression_test *= -1
        #     regression_test = (regression_test + 12) / 12
        #     descriptor_scores = [((((dd* -1)+1)/2)).tolist() for dd in descriptor_scores]
        # else:
        #     regression_test = (regression_test + 12) / 12
        #     descriptor_scores = [((dd+1)/2).tolist() for dd in descriptor_scores]


        plot_regression_outputs_by_grade_green((pred_test+1).tolist(), regression_test.tolist())
        boundaries = calculate_class_boundaries((pred_test+1).tolist(), regression_test.tolist())
        display_table_with_intervals(boundaries, split)
        plot_descriptor_scores_vs_values(descriptor_scores, X_test_scaled, columns)
        # generate_local_explainability(descriptor_scores,X_test, split, ids_test, columns, (y_test+1).tolist(), (pred_test+1).tolist())

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