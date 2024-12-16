import os
import traceback
import json
from os.path import split, splitext
from itertools import chain, combinations
from statistics import mean, stdev

from music21.tempo import MetronomeMark
from sklearn import preprocessing
from sklearn.metrics import balanced_accuracy_score, mean_squared_error

from extractor.DifficultyFeatures.code.difficulty_calculators import INSTRUMENT_DIFFICULTY_CALCULATORS
from extractor.DifficultyFeatures.code.raw_data_extractors import (
    get_default_metronome_mark,
    print_warnings,
    convert_offsets,
    CIPIFeatureExtractor
)
from rubricnet.rubricnet import RubricnetSklearn
import pandas as pd
import torch


class FakeArgs:
    """Fake arguments to initialize the RubricnetSklearn model."""
    def __init__(self):
        self.lr = 0.01
        self.batch_size = 32
        self.hidden_size = 64
        self.num_layers = 2
        self.dropout = 0.1
        self.decay_lr = 0.5
        self.weight_decay = 0.01
        self.patience = 10
        self.alias_experiment = "fake_experiment"


def extract_features(xml_path, debug=False, files_directory="tmp", ignore_warnings=True, default_metronome_mark=None):
    """
    Extracts raw data features from a music XML file.
    """
    print(f"Processing {xml_path}...")
    if debug:
        print(xml_path)

    warnings_set = set()
    try:
        default_metronome_mark = MetronomeMark(100)
        bpm_map = None
        features_extractor = CIPIFeatureExtractor()
        raw_data_map = features_extractor.extract_raw_data_map(
            [xml_path],
            warnings_set,
            metronome_mark=default_metronome_mark,
            ignore_warnings=ignore_warnings
        )
    except BaseException as err:
        if ignore_warnings:
            traceback.print_exc()
            warnings_set.add("Fatal error (nothing is written)")
            print_warnings(warnings_set)

            with open("err.log", "a") as log_file:
                print("An error occurred:", file=log_file)
                traceback.print_exc(file=log_file)
                print_warnings(warnings_set, file=log_file)
                print("", file=log_file)
        return {}

    convert_offsets(raw_data_map)
    return raw_data_map


def difficulty_descriptors(raw_data_map):
    """
    Calculates difficulty descriptors from raw data.
    """
    calculator = INSTRUMENT_DIFFICULTY_CALCULATORS['cipi']
    warnings_set = set()
    entry = calculator.estimate_difficulty_map(raw_data_map, warnings_set)

    if warnings_set:
        print_warnings(warnings_set)

    descriptors_to_return = [
        "rh_pitch_set_lz", "lh_pitch_set_lz", "rh_pitch_range", "lh_pitch_range",
        "lh_average_pitch", "rh_average_pitch", "lh_average_ioi_seconds",
        "rh_average_ioi_seconds", "rh_displacement_rate", "lh_displacement_rate",
        "lh_pitch_entropy", "rh_pitch_entropy"
    ]
    return {key: entry[key] for key in descriptors_to_return if key in entry}

def check_features_extractor():
    raw_data_map = extract_features("inven01.musicxml")
    features = difficulty_descriptors(raw_data_map)
    print("Extracted Features:", features)

    published_results = {
        "rh_pitch_set_lz": 78,
        "lh_pitch_set_lz": 76,
        "rh_pitch_range": 24.0,
        "lh_pitch_range": 34.0,
        "lh_average_pitch": 57,
        "rh_average_pitch": 72,
        "lh_average_ioi_seconds": 0.40148387096774196,
        "rh_average_ioi_seconds": 0.3745966386554622,
        "rh_displacement_rate": 0.7658227848101266,
        "lh_displacement_rate": 0.5902777777777778,
        "lh_pitch_entropy": 4.080580792251002,
        "rh_pitch_entropy": 3.7591406644141703
    }

    for key, expected_value in published_results.items():
        assert key in features, f"Key '{key}' not found in features"
        assert features[
                   key] == expected_value, f"Descriptor '{key}' value mismatch: expected {expected_value}, got {features[key]}"


def scale_to_0_12(x, min_val=-12, max_val=12):
    return (x - min_val) / (max_val - min_val) * 12


def load_model_and_predict(features):
    """
    Loads the pre-trained RubricNet model and predicts difficulty using the provided features.
    """
    os.environ["WANDB_MODE"] = "disabled"
    args = FakeArgs()
    difficulties_split = {}
    for split in range(5):
        clf = RubricnetSklearn(input_dim=len(features), num_classes=9, args=args)
        clf.load_model(f"~/PycharmProjects/Rubricnet/checkpoints/rubricnet_cameraready/split_{split}.ckpt")
        scaler = pd.read_pickle(f"~/PycharmProjects/Rubricnet/checkpoints/rubricnet_cameraready/scaler_{split}.pkl")
        scaled_features = scaler.transform([list(features.values())])
        difficulty = clf.predict(scaled_features)[0].detach().cpu().item()
        regression = clf.predict_regression(scaled_features).item()
        difficulties_split[split] = [difficulty, scale_to_0_12(regression)]
    return difficulties_split


def predict(xml_path):
    """
    Extracts features, calculates descriptors, and predicts difficulty using RubricNet in one line.
    """
    raw_representation = extract_features(xml_path)
    features = difficulty_descriptors(raw_representation)
    difficulties = load_model_and_predict(features)
    return {"features": features, "rubricnet": difficulties}


if __name__ == '__main__':
    features = predict("inven01.musicxml")
    print("Predicted Difficulty:", features)
