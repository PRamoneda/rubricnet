import os
import json
import argparse
import sys
from scipy import stats
import numpy as np
from scipy.stats import entropy
import itertools
from collections import Counter

from extractor.DifficultyFeatures.code.rhythm_chart import song_rhythm_info_from_dict
from extractor.DifficultyFeatures.code.raw_data_extractors import load_raw_data, print_warnings
from extractor.DifficultyFeatures.code.bayesian_model import keys_distance_colors

def time_signature_rank(ts):
    # TODO: reconsider naive approach to the difficulty
    ranks = {'2/4': 1, '4/4': 2, '3/4': 3, '3/8': 3, '2/2': 1}
    if ts not in ranks:
        return 4
    return ranks[ts]


def piano_time_signature_rank(ts):
    # intial: {'2/4': 1, '4/4': 1, '3/4': 2, '2/2': 2, '3/8': 3, '4/8':3, '6/8':4, '9/8':5, '12/8':6}
    ranks = {'2/4': 0, '4/4': 0, '3/4': 1, '2/2': 1, '3/8': 2, '4/8':2, '6/8':2, '9/8':3, '12/8':3}
    if ts not in ranks:
        return 3
    return ranks[ts]


def bpm_difficulty(bpm):
    """
    BPM difficulty is deviation from the "most convenient"
    """
    return abs(bpm - 110)


class DifficultyCalculator:
    """
    Interface for calculating difficulties_map from raw_data_map
    """
    def estimate_difficulty_map(self, raw_data_map, warnings_set):
        """
        Estimates difficulties map.
        Parameters
        ----------
        warnings_set
        raw_data_map
        Dictionary {<str raw_feature_id>: <JSON serializable raw data>}

        Returns
        -------
        Dictionary {<str feature_id>: float}
        """
        pass


class BaseDifficultyCalculator(DifficultyCalculator):
    """
    Estimation of the difficulties which works similarly for all instruments.
    """
    def estimate_difficulty_map(self, raw_data_map, warnings_set):
        return {
            'time_signature_rank': time_signature_rank(raw_data_map['time_signature']),
            'bpm_orig': raw_data_map['bpm_orig'],
            'bpm_ref': raw_data_map['bpm_ref'],
            'bpm': raw_data_map['bpm'],
            'bpm_dev': bpm_difficulty(raw_data_map['bpm']),
            'duration': raw_data_map['duration'],
            'unextended_duration': raw_data_map['unextended_duration'],
            'dynamics_change': raw_data_map['dynamics_change_rate'],
            'pitch_entropy': raw_data_map['pitch_entropy'],
        }


def safe_entropy(x, qk=None):
    if len(x) == 0 or max(x) == 0:
        return 0
    else:
        return stats.entropy(x, qk=qk, base=2)

def weighted_average(segment_values, segments):
    sizes_array = np.array([len(x.bars) for x in segments])
    values_array = np.array(segment_values)
    return np.sum(sizes_array * values_array / np.sum(sizes_array))

def calculate_rhythm_features(x, warnings_set, prefix = ''):
    syncopes_entropies = []
    total_entropies = []
    naive_metric_entropies = []
    syncopation_sum = 0
    syncopation_sum_salient = 0
    syncopation_weighted = 0
    syncopation_weighted_salient = 0
    song_info = song_rhythm_info_from_dict(x, warnings_set)
    # TODO: level-loop-time weighted
    for s in song_info.segments:
        s.decimate_empty_layers()
        level_durations = s.metrical_hierarchy.level_unit_durations_ms()
        position_levels = s.position_levels
        salient_positions = [level_durations[position_levels[x]] >= 100 and
                             level_durations[position_levels[x]] <= 1000
                             for x in range(len(s.syncopes_histogram))]
        syncopation_sum += sum(s.syncopes_histogram)
        syncopation_sum_salient += sum(s.syncopes_histogram[salient_positions])
        syncopation_weighted += \
            sum([x[0] * x[1] for x in zip(
                s.syncopes_histogram,
                s.syncopation_strengths)])
        syncopation_weighted_salient += \
            sum([x[0] * x[1] for x in zip(
                s.syncopes_histogram[salient_positions],
                s.syncopation_strengths[salient_positions])])
        syncopes_entropies.append(safe_entropy(s.syncopes_histogram))
        total_entropies.append(safe_entropy(
            np.concatenate((s.syncopes_histogram, s.regulars_histogram), axis=None)))
        naive_metric_entropies.append(safe_entropy(
            s.syncopes_histogram + s.regulars_histogram))

    return {prefix + 'syncopation_sum': syncopation_sum,
            prefix + 'syncopation_weighted': syncopation_weighted,
            prefix + 'syncopation_sum_salient': syncopation_sum_salient,
            prefix + 'syncopation_weighted_salient': syncopation_weighted_salient,
            prefix + 'syncopes_entropy': weighted_average(syncopes_entropies, song_info.segments),
            prefix + 'metric_entropy': weighted_average(total_entropies, song_info.segments),
            prefix + 'naive_metric_entropy': weighted_average(naive_metric_entropies, song_info.segments),
            prefix + 'tempo_signature_modes': len(x['segments'])}


def calculate_rhythm_divergence(lh_x, rh_x, warnings_set):
    syncopes_divergences = []
    total_divergences = []
    naive_metric_divergences = []
    lh_song_info = song_rhythm_info_from_dict(lh_x, warnings_set)
    rh_song_info = song_rhythm_info_from_dict(rh_x, warnings_set)
    assert len(lh_song_info.segments) == len(rh_song_info.segments)
    for i in range(len(lh_song_info.segments)):
        lh_s = lh_song_info.segments[i]
        rh_s = rh_song_info.segments[i]
        syncopes_divergences.append(safe_entropy(
            rh_s.syncopes_histogram + 1, qk=lh_s.syncopes_histogram + 1))
        total_divergences.append(safe_entropy(
            np.concatenate((rh_s.syncopes_histogram, rh_s.regulars_histogram), axis=None) + 1,
            qk=np.concatenate((lh_s.syncopes_histogram, lh_s.regulars_histogram), axis=None) + 1))
        naive_metric_divergences.append(safe_entropy(
            rh_s.syncopes_histogram + rh_s.regulars_histogram + 1,
            qk=(lh_s.syncopes_histogram + lh_s.regulars_histogram + 1)))

    return {'syncopes_divergence': weighted_average(syncopes_divergences, lh_song_info.segments),
            'metric_divergence': weighted_average(total_divergences, lh_song_info.segments),
            'naive_metric_divergence': weighted_average(naive_metric_divergences, lh_song_info.segments)}


class MonophonicDifficultyCalculator(BaseDifficultyCalculator):
    def estimate_difficulty_map(self, raw_data_map, warnings_set):
        result = super().estimate_difficulty_map(raw_data_map, warnings_set)
        result.update({
            'avg_playing_speed': raw_data_map['avg_playing_speed']['avg_playing_speed_per_score'],
            'pitch_range': raw_data_map['pitch_range'][1] - raw_data_map['pitch_range'][0]
        })
        result.update(calculate_rhythm_features(raw_data_map['song_rhythm_info'], warnings_set))
        return result


def time_signatures_info(segments):
    time_signatures = {}
    for s in segments:
        if s["metrical_hierarchy"] in time_signatures:
            time_signatures[s["metrical_hierarchy"]] += len(s["bars"])
        else:
            time_signatures[s["metrical_hierarchy"]] = len(s["bars"])

    predominant_time_signature =\
        max(time_signatures.items(), key = lambda x: x[1])[0]
    return {
        "time_signature_rank":piano_time_signature_rank(predominant_time_signature),
        "time_signature_rank_old":time_signature_rank(predominant_time_signature),
        # for compatibility! TODO: recalculate data and remove.
        "time_signature":time_signature_rank(predominant_time_signature),
        "time_signatures":time_signatures,
        "predominant_time_signature":predominant_time_signature,
        "time_signature_entropy":entropy(list(time_signatures.values()), base=2),}


def count_ornaments(o):
    # TODO: redesign raw data selection
    return sum(1 for x in o if x != 'note' and x != 'Fermata')

def uniq_values(rows):
    return np.unique(list(itertools.chain.from_iterable(rows)))

def chord_layers(rows):
    return [len(x) for x in rows if len(x) > 1 and len(x) <= 5]

def chord_spreads(rows):
    return [keys_distance_colors(min(x), max(x))[0] for x in rows if len(x) > 1  and len(x) <= 5]

def replace_nans(l):
    a = np.array(l)
    a[np.isnan(a)] = 0
    return a.tolist()

def divide_keep_valid(l1, l2):
    res = np.divide(l1, l2)
    return [x for x in res if ~np.isinf(x) and ~np.isnan(x)]

class KeysDifficultyCalculator(BaseDifficultyCalculator):
    def estimate_difficulty_map(self, raw_data_map, warnings_set):
        result = super().estimate_difficulty_map(raw_data_map, warnings_set)

        result.update({
            'pitch_set_entropy':raw_data_map['pitch_set_entropy'],
            'pitch_class_set_entropy':raw_data_map['pitch_class_set_entropy'],
            'key_pattern_entropy':raw_data_map['key_pattern_entropy'],
            'pitch_set_entropy_rate':raw_data_map['pitch_set_entropy_rate'],
            'pitch_set_lz':raw_data_map['pitch_set_lz'],
            'pitch_class_set_entropy_rate':raw_data_map['pitch_class_set_entropy_rate'],
            'key_pattern_entropy_rate':raw_data_map['key_pattern_entropy_rate'],

            'rh_pitch_set_entropy':raw_data_map['rh_pitch_set_entropy'],
            'rh_pitch_class_set_entropy':raw_data_map['rh_pitch_class_set_entropy'],
            'rh_key_pattern_entropy':raw_data_map['rh_key_pattern_entropy'],
            'rh_pitch_set_entropy_rate':raw_data_map['rh_pitch_set_entropy_rate'],
            'rh_pitch_set_lz':raw_data_map['rh_pitch_set_lz'],
            'rh_pitch_class_set_entropy_rate':raw_data_map['rh_pitch_class_set_entropy_rate'],
            'rh_key_pattern_entropy_rate':raw_data_map['rh_key_pattern_entropy_rate'],

            'lh_pitch_set_entropy':raw_data_map['lh_pitch_set_entropy'],
            'lh_pitch_class_set_entropy':raw_data_map['lh_pitch_class_set_entropy'],
            'lh_key_pattern_entropy':raw_data_map['lh_key_pattern_entropy'],
            'lh_pitch_set_entropy_rate':raw_data_map['lh_pitch_set_entropy_rate'],
            'lh_pitch_set_lz':raw_data_map['lh_pitch_set_lz'],
            'lh_pitch_class_set_entropy_rate':raw_data_map['lh_pitch_class_set_entropy_rate'],
            'lh_key_pattern_entropy_rate':raw_data_map['lh_key_pattern_entropy_rate'],

            'pitch_class_entropy':raw_data_map['pitch_class_entropy'],
            'rh_avg_playing_speed': raw_data_map['new_rh_song_rhythm_info']['avg_playing_speed_per_score'],
            'lh_avg_playing_speed': raw_data_map['new_lh_song_rhythm_info']['avg_playing_speed_per_score'],
            'rh_avg_playing_speed2': raw_data_map['new_rh_song_rhythm_info']['avg_playing_speed2_per_score'],
            'lh_avg_playing_speed2': raw_data_map['new_lh_song_rhythm_info']['avg_playing_speed2_per_score'],
            'rh_stamina': raw_data_map['new_rh_song_rhythm_info']['stamina'],
            'lh_stamina': raw_data_map['new_lh_song_rhythm_info']['stamina'],
            'joint_stamina': raw_data_map['joint_song_rhythm_info']['stamina'],
            'new_rh_stamina': raw_data_map['new_rh_song_rhythm_info']['stamina_new'],
            'new_lh_stamina': raw_data_map['new_lh_song_rhythm_info']['stamina_new'],
            'new_joint_stamina': raw_data_map['joint_song_rhythm_info']['stamina_new'],
            'rh_ioi': raw_data_map['new_rh_song_rhythm_info']['ioi_quarter_values'],
            'lh_ioi': raw_data_map['new_lh_song_rhythm_info']['ioi_quarter_values'],
            'joint_ioi': raw_data_map['joint_song_rhythm_info']['ioi_quarter_values'],
            'rh_pitch_range': raw_data_map['rh_pitch_range'][1] - raw_data_map['rh_pitch_range'][0],
            'lh_pitch_range': raw_data_map['lh_pitch_range'][1] - raw_data_map['lh_pitch_range'][0],
            "articulations" : raw_data_map['articulations']['articulations_entropy'],
            # Part Writing
            "avg_chord_density" : raw_data_map['avg_chord_density'],
            "max_chord_density" : raw_data_map['max_chord_density'],
            "avg_chord_spread" : raw_data_map['avg_chord_spread'],
            "rh_pitch_intervals": raw_data_map['rh_pitch_intervals'],
            "rh_time_intervals": raw_data_map['rh_time_intervals'],
            "rh_pitch_sets": raw_data_map['rh_pitch_sets'],
            "rh_finger_sets": raw_data_map['rh_finger_sets'],
            "lh_pitch_intervals": raw_data_map['lh_pitch_intervals'],
            "lh_time_intervals": raw_data_map['lh_time_intervals'],
            "lh_pitch_sets": raw_data_map['lh_pitch_sets'],
            "lh_finger_sets": raw_data_map['lh_finger_sets'],
            "rh_ornaments": raw_data_map['rh_ornaments'],
            "lh_ornaments": raw_data_map['lh_ornaments'],
            "rh_clefs": raw_data_map['rh_clefs'],
            "lh_clefs": raw_data_map['lh_clefs'],
            "key_signatures": raw_data_map['key_signatures'],
            # rhythm/tempo
            "tuplets": raw_data_map['tuplets']
        })
        result['lh_average_ioi_seconds'] = 1.0 / result['lh_avg_playing_speed']
        result['rh_average_ioi_seconds'] = 1.0 / result['rh_avg_playing_speed']
        # Time signatures (from segments)
        result.update(time_signatures_info(raw_data_map['lh_song_rhythm_info']["segments"]))

        # predominant_time_signature
        # time_signature_entropy
        # time_signatures

        result.update(calculate_rhythm_features(raw_data_map['lh_song_rhythm_info'], warnings_set, prefix='lh_'))
        result.update(calculate_rhythm_features(raw_data_map['rh_song_rhythm_info'], warnings_set, prefix='rh_'))
        result.update(calculate_rhythm_features(raw_data_map['new_lh_song_rhythm_info'], warnings_set, prefix='new_lh_'))
        result.update(calculate_rhythm_features(raw_data_map['new_rh_song_rhythm_info'], warnings_set, prefix='new_rh_'))
        result.update(calculate_rhythm_features(raw_data_map['joint_song_rhythm_info'], warnings_set, prefix='joint_'))
        # result.update(calculate_rhythm_features(raw_data_map['song_rhythm_info'], warnings_set, prefix=''))
        result['sum_metric_entropy'] = result['rh_metric_entropy'] + result['lh_metric_entropy']
        result['sum_syncopation_sum'] = result['rh_syncopation_sum'] + result['lh_syncopation_sum']
        result.update(calculate_rhythm_divergence(
            raw_data_map['lh_song_rhythm_info'],
            raw_data_map['rh_song_rhythm_info'], warnings_set))
        result['tempo_signature_modes'] = result['lh_tempo_signature_modes']
        result.pop('lh_tempo_signature_modes')
        result.pop('rh_tempo_signature_modes')

        result['rh_horizontal_velocities'] = divide_keep_valid(result['rh_pitch_intervals'], result['rh_time_intervals'])
        result['rh_horizontal_velocities2'] = np.multiply(result['rh_horizontal_velocities'], result['rh_horizontal_velocities']).tolist()
        result['lh_horizontal_velocities'] = divide_keep_valid(result['lh_pitch_intervals'], result['lh_time_intervals'])
        result['lh_horizontal_velocities2'] = np.multiply(result['lh_horizontal_velocities'], result['lh_horizontal_velocities']).tolist()

        result["rh_avg_time_interval"] = 1000.0 / np.mean(result['rh_time_intervals'])
        result["rh_avg_pitch_interval"] = np.mean(np.abs(result['rh_pitch_intervals']))
        result["rh_avg_horizontal_velocities"] = np.mean(np.abs(result['rh_horizontal_velocities']))
        result["rh_avg_horizontal_velocities2"] = np.mean(result['rh_horizontal_velocities2'])
        if np.isnan(result["rh_avg_horizontal_velocities2"]):
            result["rh_avg_horizontal_velocities2"] = 0

        result["lh_avg_time_interval"] = replace_nans([np.mean(np.abs(x)) for x in result['lh_time_intervals']])
        result["lh_avg_pitch_interval"] = replace_nans([np.mean(np.abs(x)) for x in result['lh_pitch_intervals']])
        result["lh_avg_horizontal_velocities"] = replace_nans([np.mean(np.abs(x)) for x in result['lh_horizontal_velocities']])
        result["lh_avg_horizontal_velocities2"] = np.mean(result['lh_horizontal_velocities2'])
        if np.isnan(result["lh_avg_horizontal_velocities2"]):
            result["lh_avg_horizontal_velocities2"] = 0

        result['rh_clef_changes'] = len(result['rh_clefs'])
        result['lh_clef_changes'] = len(result['lh_clefs'])

        result['lh_chord_layers'] = chord_layers(result['lh_pitch_sets'])
        result['lh_chord_spreads'] = chord_spreads(result['lh_pitch_sets'])
        result['rh_chord_layers'] = chord_layers(result['rh_pitch_sets'])
        result['rh_chord_spreads'] = chord_spreads(result['rh_pitch_sets'])
        result['chord_layers'] = np.concatenate((result['rh_chord_layers'], result['lh_chord_layers'])).tolist()
        result['chord_spreads'] = np.concatenate((result['rh_chord_spreads'], result['lh_chord_spreads'])).tolist()

        result["rh_ornaments_num"] = count_ornaments(result["rh_ornaments"])
        result["lh_ornaments_num"] = count_ornaments(result["lh_ornaments"])
        result["ornaments_num"] = result["rh_ornaments_num"] + result["lh_ornaments_num"]

        # To allow Poisson
        result["tuplets"] = int(100 * result["tuplets"])

        # To match with Chiu
        result['pitch_range'] = max(raw_data_map['rh_pitch_range'][1], raw_data_map['lh_pitch_range'][1]) -\
                                min(raw_data_map['rh_pitch_range'][0], raw_data_map['lh_pitch_range'][0])
        result['lh_average_pitch'] = average_pitch(raw_data_map["lh_pitch_sets"])
        result['rh_average_pitch'] = average_pitch(raw_data_map["rh_pitch_sets"])
        result['average_pitch'] = average_pitch(
            raw_data_map["lh_pitch_sets"] + raw_data_map["rh_pitch_sets"])
        result.update(displacement_rates(raw_data_map))
        result['lh_pitch_stuff'] = entropy(list(itertools.chain.from_iterable(raw_data_map["lh_pitch_sets"])), base=2)
        result['rh_pitch_stuff'] = entropy(list(itertools.chain.from_iterable(raw_data_map["rh_pitch_sets"])), base=2)
        result['new_pitch_stuff'] = entropy(
            list(itertools.chain.from_iterable(raw_data_map["rh_pitch_sets"])) +
            list(itertools.chain.from_iterable(raw_data_map["lh_pitch_sets"])), base=2)
        result['old_pitch_entropy'] = raw_data_map["old_pitch_entropy"]['pitch_entropy']
        result['lh_pitch_entropy'] = entropy(list(Counter(itertools.chain.from_iterable(raw_data_map["lh_pitch_sets"])).values()), base=2)
        result['rh_pitch_entropy'] = entropy(list(Counter(itertools.chain.from_iterable(raw_data_map["rh_pitch_sets"])).values()), base=2)
        result['new_pitch_entropy'] = entropy(
            list(Counter(list(itertools.chain.from_iterable(raw_data_map["rh_pitch_sets"])) +
                    list(itertools.chain.from_iterable(raw_data_map["lh_pitch_sets"]))).values()), base=2)
        result['rh_displacement_stuff'] = displacement_stuff(raw_data_map["rh_pitch_sets"])
        result['lh_displacement_stuff'] = displacement_stuff(raw_data_map["lh_pitch_sets"])
        result['displacement_stuff'] = result['rh_displacement_rate'] + result['lh_displacement_rate']
        return result


def average_pitch(pitch_sets):
    pitches = list(itertools.chain.from_iterable(pitch_sets))
    ap = np.mean(pitches)
    if np.isnan(ap):
        return ap
    else:
        return int(ap)

def displacement_stuff(pitch_sets):
    prev = [min(pitch_sets[0]), max(pitch_sets[1])]
    sum = 0
    for s in pitch_sets[1:]:
        cur = [min(s), max(s)]
        d = max(np.abs([x[0] - x[1] for x in itertools.product(prev, cur)]))
        d = 2 if d >= 12 else 1 if d >= 7 else 0
        sum += d
    return sum

def displacement_numerator(pitch_sets):
    sum = 0
    if (len(pitch_sets) > 1):
        prev = [min(pitch_sets[0]), max(pitch_sets[1])]
        for s in pitch_sets[1:]:
            cur = [min(s), max(s)]
            d = max(np.abs([x[0] - x[1] for x in itertools.product(prev, cur)]))
            d = 2 if d >= 12 else 1 if d >= 7 else 0
            sum += d
    return sum

def displacement_rates(raw_data_map):
    result = {}
    rh_numerator = displacement_numerator(raw_data_map["rh_pitch_sets"])
    lh_numerator = displacement_numerator(raw_data_map["lh_pitch_sets"])
    rh_intervals = len(raw_data_map["rh_pitch_sets"]) - 1
    lh_intervals = len(raw_data_map["lh_pitch_sets"]) - 1
    all_intervals = rh_intervals + lh_intervals - 2
    result['rh_displacement_rate'] =  rh_numerator / \
                                      (2 * (rh_intervals if rh_intervals > 0 else 1))
    result['lh_displacement_rate'] =  lh_numerator / \
                                      (2 * (lh_intervals if lh_intervals > 0 else 1))
    result['displacement_rate'] = (rh_numerator + lh_numerator) / \
                                  (2 * (all_intervals if all_intervals > 0 else 1))
    return result

INSTRUMENT_DIFFICULTY_CALCULATORS = {
    'guitar':  MonophonicDifficultyCalculator(),
    'vocal': MonophonicDifficultyCalculator(),
    'keys': KeysDifficultyCalculator(),
    'piano': KeysDifficultyCalculator(),
    'cipi': KeysDifficultyCalculator(),
    'bass': MonophonicDifficultyCalculator()
}

def get_args():
    """
    Parse arguments (if it's invoked as a command).
    """
    parser = argparse.ArgumentParser(
        description='Estimate score feature difficulties and store it to json file.')
    parser.add_argument(
        '-i', '--instrument', type=str, help='instrumrnt name: ' + ",".join(INSTRUMENT_DIFFICULTY_CALCULATORS.keys()), required=True)
    parser.add_argument(
        '-o', '--output', type=str, default='output.json', help='output file in .json format (default is output.json)',
        required=False)
    parser.add_argument(
        '-r', '--raw-data', type=str, help='.json with collected features raw_data (from raw_data_extractors.py)',
        required=True)
    parser.add_argument(
        '-d', '--debug', action='store_true', default=False, help='Debug mode',
        required=False)
    parser.add_argument(
        '-f', '--files', action='store_true', default=False, help='Write individual files',
        required=False)
    args = parser.parse_args()
    instrument = args.instrument
    raw_data_file = args.raw_data
    outfile = args.output
    debug = args.debug
    files = args.files
    return instrument, raw_data_file, outfile, debug, files

##############################################


if __name__ == '__main__':
    instrument, raw_data_file, out_file, debug, files = get_args()
    if files:
        # individual files in the directory
        raw_data_maps = []
        for file in os.listdir(raw_data_file):
            if file.endswith(".json"):
                with open(os.path.join(raw_data_file, file)) as f:
                    raw_data_maps.append(load_raw_data(f))
    else:
        # a single json file
        with open(raw_data_file) as f:
            raw_data_maps = load_raw_data(f)

    calculator = INSTRUMENT_DIFFICULTY_CALCULATORS[instrument]

    with open(out_file, 'w') as fo:
        print("Writing %s..." % out_file)
        res = []
        for raw_data_map in raw_data_maps:
            print(raw_data_map['id'])
            warnings_set = set()
            entry = calculator.estimate_difficulty_map(raw_data_map, warnings_set)
            entry['id'] = raw_data_map['id']
            entry['grade'] = raw_data_map['grade']
            res.append(entry)
            if len(warnings_set) > 0:
                print_warnings(warnings_set)
        json.dump(res, fo, indent=True)
