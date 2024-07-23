import json
from itertools import chain, combinations
from statistics import mean, stdev


from sklearn.metrics import balanced_accuracy_score, mean_squared_error


from rubricnet import RubricnetSklearn

import optuna

from sklearn import preprocessing


import utils

import pandas as pd



class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)


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

basic_features = ['rh_pitch_set_lz',
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

jsymbolic_features = ['Number of Pitches', 'Number of Pitch Classes', 'Number of Common Pitches', 'Number of Common Pitch Classes', 'Range', 'Importance of Bass Register', 'Importance of Middle Register', 'Importance of High Register', 'Dominant Spread', 'Strong Tonal Centres', 'Mean Pitch', 'Mean Pitch Class', 'Most Common Pitch', 'Most Common Pitch Class', 'Prevalence of Most Common Pitch', 'Prevalence of Most Common Pitch Class', 'Relative Prevalence of Top Pitches', 'Relative Prevalence of Top Pitch Classes', 'Interval Between Most Prevalent Pitches', 'Interval Between Most Prevalent Pitch Classes', 'Pitch Variability', 'Pitch Class Variability', 'Pitch Class Variability After Folding', 'Pitch Skewness', 'Pitch Class Skewness', 'Pitch Class Skewness After Folding', 'Pitch Kurtosis', 'Pitch Class Kurtosis', 'Pitch Class Kurtosis After Folding', 'Major or Minor', 'First Pitch', 'First Pitch Class', 'Last Pitch', 'Last Pitch Class', 'Glissando Prevalence', 'Average Range of Glissandos', 'Vibrato Prevalence', 'Microtone Prevalence', 'Most Common Melodic Interval', 'Mean Melodic Interval', 'Number of Common Melodic Intervals', 'Distance Between Most Prevalent Melodic Intervals', 'Prevalence of Most Common Melodic Interval', 'Relative Prevalence of Most Common Melodic Intervals', 'Amount of Arpeggiation', 'Repeated Notes', 'Chromatic Motion', 'Stepwise Motion', 'Melodic Thirds', 'Melodic Perfect Fourths', 'Melodic Tritones', 'Melodic Perfect Fifths', 'Melodic Sixths', 'Melodic Sevenths', 'Melodic Octaves', 'Melodic Large Intervals', 'Minor Major Melodic Third Ratio', 'Melodic Embellishments', 'Direction of Melodic Motion', 'Average Length of Melodic Arcs', 'Average Interval Spanned by Melodic Arcs', 'Melodic Pitch Variety', 'Average Number of Simultaneous Pitch Classes', 'Variability of Number of Simultaneous Pitch Classes', 'Average Number of Simultaneous Pitches', 'Variability of Number of Simultaneous Pitches', 'Most Common Vertical Interval', 'Second Most Common Vertical Interval', 'Distance Between Two Most Common Vertical Intervals', 'Prevalence of Most Common Vertical Interval', 'Prevalence of Second Most Common Vertical Interval', 'Prevalence Ratio of Two Most Common Vertical Intervals', 'Vertical Unisons', 'Vertical Minor Seconds', 'Vertical Thirds', 'Vertical Tritones', 'Vertical Perfect Fourths', 'Vertical Perfect Fifths', 'Vertical Sixths', 'Vertical Sevenths', 'Vertical Octaves', 'Perfect Vertical Intervals', 'Vertical Dissonance Ratio', 'Vertical Minor Third Prevalence', 'Vertical Major Third Prevalence', 'Chord Duration', 'Partial Chords', 'Standard Triads', 'Diminished and Augmented Triads', 'Dominant Seventh Chords', 'Seventh Chords', 'Non-Standard Chords', 'Complex Chords', 'Minor Major Triad Ratio', 'Simple Initial Meter', 'Compound Initial Meter', 'Complex Initial Meter', 'Duple Initial Meter', 'Triple Initial Meter', 'Quadruple Initial Meter', 'Metrical Diversity', 'Total Number of Notes', 'Note Density per Quarter Note', 'Note Density per Quarter Note per Voice', 'Note Density per Quarter Note Variability', 'Range of Rhythmic Values', 'Number of Different Rhythmic Values Present', 'Number of Common Rhythmic Values Present', 'Prevalence of Very Short Rhythmic Values', 'Prevalence of Short Rhythmic Values', 'Prevalence of Medium Rhythmic Values', 'Prevalence of Long Rhythmic Values', 'Prevalence of Very Long Rhythmic Values', 'Prevalence of Dotted Notes', 'Shortest Rhythmic Value', 'Longest Rhythmic Value', 'Mean Rhythmic Value', 'Most Common Rhythmic Value', 'Prevalence of Most Common Rhythmic Value', 'Relative Prevalence of Most Common Rhythmic Values', 'Difference Between Most Common Rhythmic Values', 'Rhythmic Value Variability', 'Rhythmic Value Skewness', 'Rhythmic Value Kurtosis', 'Mean Rhythmic Value Run Length', 'Median Rhythmic Value Run Length', 'Variability in Rhythmic Value Run Lengths', 'Mean Rhythmic Value Offset', 'Median Rhythmic Value Offset', 'Variability of Rhythmic Value Offsets', 'Complete Rests Fraction', 'Partial Rests Fraction', 'Average Rest Fraction Across Voices', 'Longest Complete Rest', 'Longest Partial Rest', 'Mean Complete Rest Duration', 'Mean Partial Rest Duration', 'Median Complete Rest Duration', 'Median Partial Rest Duration', 'Variability of Complete Rest Durations', 'Variability of Partial Rest Durations', 'Variability Across Voices of Combined Rests', 'Number of Strong Rhythmic Pulses - Tempo Standardized', 'Number of Moderate Rhythmic Pulses - Tempo Standardized', 'Number of Relatively Strong Rhythmic Pulses - Tempo Standardized', 'Strongest Rhythmic Pulse - Tempo Standardized', 'Second Strongest Rhythmic Pulse - Tempo Standardized', 'Harmonicity of Two Strongest Rhythmic Pulses - Tempo Standardized', 'Strength of Strongest Rhythmic Pulse - Tempo Standardized', 'Strength of Second Strongest Rhythmic Pulse - Tempo Standardized', 'Strength Ratio of Two Strongest Rhythmic Pulses - Tempo Standardized', 'Combined Strength of Two Strongest Rhythmic Pulses - Tempo Standardized', 'Rhythmic Variability - Tempo Standardized', 'Rhythmic Looseness - Tempo Standardized', 'Polyrhythms - Tempo Standardized', 'Initial Tempo', 'Mean Tempo', 'Tempo Variability', 'Duration in Seconds', 'Note Density', 'Note Density Variability', 'Average Time Between Attacks', 'Average Time Between Attacks for Each Voice', 'Variability of Time Between Attacks', 'Average Variability of Time Between Attacks for Each Voice', 'Minimum Note Duration', 'Maximum Note Duration', 'Average Note Duration', 'Variability of Note Durations', 'Amount of Staccato', 'Number of Strong Rhythmic Pulses', 'Number of Moderate Rhythmic Pulses', 'Number of Relatively Strong Rhythmic Pulses', 'Strongest Rhythmic Pulse', 'Second Strongest Rhythmic Pulse', 'Harmonicity of Two Strongest Rhythmic Pulses', 'Strength of Strongest Rhythmic Pulse', 'Strength of Second Strongest Rhythmic Pulse', 'Strength Ratio of Two Strongest Rhythmic Pulses', 'Combined Strength of Two Strongest Rhythmic Pulses', 'Rhythmic Variability', 'Rhythmic Looseness', 'Polyrhythms', 'Variability of Note Prevalence of Pitched Instruments', 'Variability of Note Prevalence of Unpitched Instruments', 'Number of Pitched Instruments', 'Number of Unpitched Instruments', 'Unpitched Percussion Instrument Prevalence', 'String Keyboard Prevalence', 'Acoustic Guitar Prevalence', 'Electric Guitar Prevalence', 'Violin Prevalence', 'Saxophone Prevalence', 'Brass Prevalence', 'Woodwinds Prevalence', 'Orchestral Strings Prevalence', 'String Ensemble Prevalence', 'Electric Instrument Prevalence', 'Maximum Number of Independent Voices', 'Average Number of Independent Voices', 'Variability of Number of Independent Voices', 'Voice Equality - Number of Notes', 'Voice Equality - Note Duration', 'Voice Equality - Dynamics', 'Voice Equality - Melodic Leaps', 'Voice Equality - Range', 'Importance of Loudest Voice', 'Relative Range of Loudest Voice', 'Relative Range Isolation of Loudest Voice', 'Relative Range of Highest Line', 'Relative Note Density of Highest Line', 'Relative Note Durations of Lowest Line', 'Relative Size of Melodic Intervals in Lowest Line', 'Voice Overlap', 'Voice Separation', 'Variability of Voice Separation', 'Parallel Motion', 'Similar Motion', 'Contrary Motion', 'Oblique Motion', 'Parallel Fifths', 'Parallel Octaves', 'Dynamic Range', 'Variation of Dynamics', 'Variation of Dynamics In Each Voice', 'Average Note to Note Change in Dynamics']

music21_features = ['M1_0', 'M1_1', 'M1_2', 'M1_3', 'M1_4', 'M1_5', 'M1_6', 'M1_7', 'M1_8', 'M1_9', 'M1_10', 'M1_11', 'M1_12', 'M1_13', 'M1_14', 'M1_15', 'M1_16', 'M1_17', 'M1_18', 'M1_19', 'M1_20', 'M1_21', 'M1_22', 'M1_23', 'M1_24', 'M1_25', 'M1_26', 'M1_27', 'M1_28', 'M1_29', 'M1_30', 'M1_31', 'M1_32', 'M1_33', 'M1_34', 'M1_35', 'M1_36', 'M1_37', 'M1_38', 'M1_39', 'M1_40', 'M1_41', 'M1_42', 'M1_43', 'M1_44', 'M1_45', 'M1_46', 'M1_47', 'M1_48', 'M1_49', 'M1_50', 'M1_51', 'M1_52', 'M1_53', 'M1_54', 'M1_55', 'M1_56', 'M1_57', 'M1_58', 'M1_59', 'M1_60', 'M1_61', 'M1_62', 'M1_63', 'M1_64', 'M1_65', 'M1_66', 'M1_67', 'M1_68', 'M1_69', 'M1_70', 'M1_71', 'M1_72', 'M1_73', 'M1_74', 'M1_75', 'M1_76', 'M1_77', 'M1_78', 'M1_79', 'M1_80', 'M1_81', 'M1_82', 'M1_83', 'M1_84', 'M1_85', 'M1_86', 'M1_87', 'M1_88', 'M1_89', 'M1_90', 'M1_91', 'M1_92', 'M1_93', 'M1_94', 'M1_95', 'M1_96', 'M1_97', 'M1_98', 'M1_99', 'M1_100', 'M1_101', 'M1_102', 'M1_103', 'M1_104', 'M1_105', 'M1_106', 'M1_107', 'M1_108', 'M1_109', 'M1_110', 'M1_111', 'M1_112', 'M1_113', 'M1_114', 'M1_115', 'M1_116', 'M1_117', 'M1_118', 'M1_119', 'M1_120', 'M1_121', 'M1_122', 'M1_123', 'M1_124', 'M1_125', 'M1_126', 'M1_127', 'M2_0', 'M3_0', 'M4_0', 'M5_0', 'M6_0', 'M7_0', 'M8_0', 'M9_0', 'm10_0', 'M11_0', 'M12_0', 'M13_0', 'M14_0', 'M15_0', 'm17_0', 'M18_0', 'M19_0', 'I1_0', 'I1_1', 'I1_2', 'I1_3', 'I1_4', 'I1_5', 'I1_6', 'I1_7', 'I1_8', 'I1_9', 'I1_10', 'I1_11', 'I1_12', 'I1_13', 'I1_14', 'I1_15', 'I1_16', 'I1_17', 'I1_18', 'I1_19', 'I1_20', 'I1_21', 'I1_22', 'I1_23', 'I1_24', 'I1_25', 'I1_26', 'I1_27', 'I1_28', 'I1_29', 'I1_30', 'I1_31', 'I1_32', 'I1_33', 'I1_34', 'I1_35', 'I1_36', 'I1_37', 'I1_38', 'I1_39', 'I1_40', 'I1_41', 'I1_42', 'I1_43', 'I1_44', 'I1_45', 'I1_46', 'I1_47', 'I1_48', 'I1_49', 'I1_50', 'I1_51', 'I1_52', 'I1_53', 'I1_54', 'I1_55', 'I1_56', 'I1_57', 'I1_58', 'I1_59', 'I1_60', 'I1_61', 'I1_62', 'I1_63', 'I1_64', 'I1_65', 'I1_66', 'I1_67', 'I1_68', 'I1_69', 'I1_70', 'I1_71', 'I1_72', 'I1_73', 'I1_74', 'I1_75', 'I1_76', 'I1_77', 'I1_78', 'I1_79', 'I1_80', 'I1_81', 'I1_82', 'I1_83', 'I1_84', 'I1_85', 'I1_86', 'I1_87', 'I1_88', 'I1_89', 'I1_90', 'I1_91', 'I1_92', 'I1_93', 'I1_94', 'I1_95', 'I1_96', 'I1_97', 'I1_98', 'I1_99', 'I1_100', 'I1_101', 'I1_102', 'I1_103', 'I1_104', 'I1_105', 'I1_106', 'I1_107', 'I1_108', 'I1_109', 'I1_110', 'I1_111', 'I1_112', 'I1_113', 'I1_114', 'I1_115', 'I1_116', 'I1_117', 'I1_118', 'I1_119', 'I1_120', 'I1_121', 'I1_122', 'I1_123', 'I1_124', 'I1_125', 'I1_126', 'I1_127', 'I3_0', 'I3_1', 'I3_2', 'I3_3', 'I3_4', 'I3_5', 'I3_6', 'I3_7', 'I3_8', 'I3_9', 'I3_10', 'I3_11', 'I3_12', 'I3_13', 'I3_14', 'I3_15', 'I3_16', 'I3_17', 'I3_18', 'I3_19', 'I3_20', 'I3_21', 'I3_22', 'I3_23', 'I3_24', 'I3_25', 'I3_26', 'I3_27', 'I3_28', 'I3_29', 'I3_30', 'I3_31', 'I3_32', 'I3_33', 'I3_34', 'I3_35', 'I3_36', 'I3_37', 'I3_38', 'I3_39', 'I3_40', 'I3_41', 'I3_42', 'I3_43', 'I3_44', 'I3_45', 'I3_46', 'I3_47', 'I3_48', 'I3_49', 'I3_50', 'I3_51', 'I3_52', 'I3_53', 'I3_54', 'I3_55', 'I3_56', 'I3_57', 'I3_58', 'I3_59', 'I3_60', 'I3_61', 'I3_62', 'I3_63', 'I3_64', 'I3_65', 'I3_66', 'I3_67', 'I3_68', 'I3_69', 'I3_70', 'I3_71', 'I3_72', 'I3_73', 'I3_74', 'I3_75', 'I3_76', 'I3_77', 'I3_78', 'I3_79', 'I3_80', 'I3_81', 'I3_82', 'I3_83', 'I3_84', 'I3_85', 'I3_86', 'I3_87', 'I3_88', 'I3_89', 'I3_90', 'I3_91', 'I3_92', 'I3_93', 'I3_94', 'I3_95', 'I3_96', 'I3_97', 'I3_98', 'I3_99', 'I3_100', 'I3_101', 'I3_102', 'I3_103', 'I3_104', 'I3_105', 'I3_106', 'I3_107', 'I3_108', 'I3_109', 'I3_110', 'I3_111', 'I3_112', 'I3_113', 'I3_114', 'I3_115', 'I3_116', 'I3_117', 'I3_118', 'I3_119', 'I3_120', 'I3_121', 'I3_122', 'I3_123', 'I3_124', 'I3_125', 'I3_126', 'I3_127', 'I6_0', 'I8_0', 'I11_0', 'I12_0', 'I13_0', 'I14_0', 'I15_0', 'I16_0', 'I17_0', 'I18_0', 'I19_0', 'I20_0', 'R15_0', 'R17_0', 'R18_0', 'R19_0', 'R20_0', 'R21_0', 'R22_0', 'R23_0', 'R24_0', 'R25_0', 'R30_0', 'R31_0', 'R31_1', 'R32_0', 'R33_0', 'R34_0', 'R35_0', 'R36_0', 'T1_0', 'T2_0', 'T3_0', 'P1_0', 'P2_0', 'P3_0', 'P4_0', 'P5_0', 'P6_0', 'P7_0', 'P8_0', 'P9_0', 'P10_0', 'P11_0', 'P12_0', 'P13_0', 'P14_0', 'P15_0', 'P16_0', 'P19_0', 'P19_1', 'P19_2', 'P19_3', 'P19_4', 'P19_5', 'P19_6', 'P19_7', 'P19_8', 'P19_9', 'P19_10', 'P19_11', 'P19_12', 'P19_13', 'P19_14', 'P19_15', 'P19_16', 'P19_17', 'P19_18', 'P19_19', 'P19_20', 'P19_21', 'P19_22', 'P19_23', 'P19_24', 'P19_25', 'P19_26', 'P19_27', 'P19_28', 'P19_29', 'P19_30', 'P19_31', 'P19_32', 'P19_33', 'P19_34', 'P19_35', 'P19_36', 'P19_37', 'P19_38', 'P19_39', 'P19_40', 'P19_41', 'P19_42', 'P19_43', 'P19_44', 'P19_45', 'P19_46', 'P19_47', 'P19_48', 'P19_49', 'P19_50', 'P19_51', 'P19_52', 'P19_53', 'P19_54', 'P19_55', 'P19_56', 'P19_57', 'P19_58', 'P19_59', 'P19_60', 'P19_61', 'P19_62', 'P19_63', 'P19_64', 'P19_65', 'P19_66', 'P19_67', 'P19_68', 'P19_69', 'P19_70', 'P19_71', 'P19_72', 'P19_73', 'P19_74', 'P19_75', 'P19_76', 'P19_77', 'P19_78', 'P19_79', 'P19_80', 'P19_81', 'P19_82', 'P19_83', 'P19_84', 'P19_85', 'P19_86', 'P19_87', 'P19_88', 'P19_89', 'P19_90', 'P19_91', 'P19_92', 'P19_93', 'P19_94', 'P19_95', 'P19_96', 'P19_97', 'P19_98', 'P19_99', 'P19_100', 'P19_101', 'P19_102', 'P19_103', 'P19_104', 'P19_105', 'P19_106', 'P19_107', 'P19_108', 'P19_109', 'P19_110', 'P19_111', 'P19_112', 'P19_113', 'P19_114', 'P19_115', 'P19_116', 'P19_117', 'P19_118', 'P19_119', 'P19_120', 'P19_121', 'P19_122', 'P19_123', 'P19_124', 'P19_125', 'P19_126', 'P19_127', 'P20_0', 'P20_1', 'P20_2', 'P20_3', 'P20_4', 'P20_5', 'P20_6', 'P20_7', 'P20_8', 'P20_9', 'P20_10', 'P20_11', 'P21_0', 'P21_1', 'P21_2', 'P21_3', 'P21_4', 'P21_5', 'P21_6', 'P21_7', 'P21_8', 'P21_9', 'P21_10', 'P21_11', 'P22_0', 'K1_0', 'QL1_0', 'QL2_0', 'QL3_0', 'QL4_0', 'CS1_0', 'CS2_0', 'CS3_0', 'CS4_0', 'CS5_0', 'CS6_0', 'CS7_0', 'CS8_0', 'CS9_0', 'CS10_0', 'CS11_0', 'CS12_0', 'CS12_1', 'CS12_2', 'CS12_3', 'CS12_4', 'CS12_5', 'CS12_6', 'CS12_7', 'CS12_8', 'CS12_9', 'CS12_10', 'CS12_11', 'MC1_0', 'TX1_0']

def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(1, len(ss)+1)))


def main(trial):
    if FEATURES == "jsymbolic":
        columns = jsymbolic_features
    elif FEATURES == "music21":
        columns = music21_features
    elif FEATURES == "all":
        columns = music21_features + basic_features
    elif FEATURES == "basic":
        columns = basic_features
    elif FEATURES == "chiu":
        columns = minimal_columns_chiu
    else:
        raise ValueError("Invalid FEATURES")
    dataset = "cipi"
    # dataset = "mikrokosmos"
    args = {
        "batch_size": trial.suggest_int('batch_size', 16, 128),
        "patience": 50,
        "alias_experiment": f"{trial.study.study_name}_" + str(trial.number),
        "weight_decay": 0.01,
        "hidden_size": 1,
        "num_layers": 1,
        "dropout": trial.suggest_float('dropout', 0.3, 0.5),
        "decay_lr": trial.suggest_float('decay_lr', 0.1, 0.9),
        "lr": trial.suggest_float('lr', 1e-5, 1e-1, log=True),
    }
    args = Args(**args)
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
    else:
        data = utils.load_json("new_splits.json")
        features = {v["id"]: v for v in utils.load_json("../features/mikrokosmos_difficulties.json")}

    final_acc9_val, final_acc9_test, final_mse_val, final_mse_test = [], [], [], []
    best_acc_val, best_selection = 0, ""

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
                                          args=args, logging=False)
        clf.fit(scaler.transform(X_train), y_train, scaler.transform(X_val), y_val, scaler.transform(X_test), y_test)

        # load from checkpoint
        clf.load_model(f"checkpoints/{args.alias_experiment}/split_{split}.ckpt")
        pred_val = clf.predict(scaler.transform(X_val))
        pred_test = clf.predict(scaler.transform(X_test))

        acc9_val = balanced_accuracy_score(y_true=y_val, y_pred=pred_val)
        acc9_test = balanced_accuracy_score(y_true=y_test, y_pred=pred_test)

        mse_val = get_mse_macro(y_true=y_val, y_pred=pred_val)
        mse_test = get_mse_macro(y_true=y_test, y_pred=pred_test)
        final_acc9_test.append(acc9_test)
        final_acc9_val.append(acc9_val)
        final_mse_val.append(mse_val)
        final_mse_test.append(mse_test)
    trial.set_user_attr("acc9_test", mean(final_acc9_test))
    trial.set_user_attr("acc9_test_st", stdev(final_acc9_test))
    trial.set_user_attr("acc9_val", mean(final_acc9_val))
    trial.set_user_attr("mse_val", mean(final_mse_val))
    trial.set_user_attr("mse_test", mean(final_mse_test))
    trial.set_user_attr("mse_test_st", stdev(final_mse_test))
    return mean(final_acc9_val), mean(final_mse_val)


FEATURES = "basic"


if __name__ == '__main__':
    alias_optuna = "basic"
    sqlite_url = f'sqlite:///{alias_optuna}.db'
    study = optuna.create_study(study_name=alias_optuna, directions=['maximize', 'minimize'], storage=sqlite_url, load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=2))
    study.optimize(main, n_trials=99, n_jobs=1)
    print('Best trial:')
    trial = study.best_trial
    print(f'  Accuracy: {trial.value}')
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')