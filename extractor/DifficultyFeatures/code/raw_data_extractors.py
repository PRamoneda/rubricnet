import numpy as np
from music21 import stream
import argparse
import json
import os
import sys
from copy import deepcopy
import traceback
from os.path import exists
from music21.tempo import MetronomeMark
import glob

from extractor.DifficultyFeatures.code.performance_chain import pitch_distributions, performance_chain
from extractor.DifficultyFeatures.code.rhythm_chart import syncopation_statistics, quarter_ms, create_segment_by_meter_tempo, \
    rhythm_segments
from extractor.DifficultyFeatures.code.rhythm_chart_new import syncopation_statistics_for_part, syncopation_statistics_for_lists
from extractor.DifficultyFeatures.code.common_utils import safe_convert, rhythm_tempo_features, \
    pitch_range_per_part, dynamics_general_features, \
    pitch_entropy, playing_speed_per_s_per_part, get_measure_offsets, get_song_map, \
    propagate_metr_events_to_parts, extract_metr_events_and_first_measure_number, \
    build_offset_to_effective_bar_number_map, ornaments_statistics
from extractor.DifficultyFeatures.code.keys_features import articulation_features, tuplets_ratio, chord_density, avg_chord_spread
from piano_fingering_argnn.compute_embeddings import compute_piece_argnn
from extractor.DifficultyFeatures.code.clefs_signatures import clefs_statistics, key_signatures_statistics

class MissedMetronomeMarkException(Exception):
    pass


class FeaturesExtractor:
    """
    Interface for raw data extraction from a single musicxml file.
    """

    def extract_raw_data_map_from_streams(
            self,
            m21_stream,
            m21_stream_with_metronome_marks,
            m21_expanded_stream_with_metronome_marks,
            warnings_set):
        pass

    def extract_raw_data_map(self, music_xml_filenames, warnings_set, metronome_mark = None, ignore_warnings = False):
        """
        Extracts features raw data from music xml.

        Parameters
        ----------
        music_xml_filename

        Returns
        -------
        Dictionary {<str feature_id>: <JSON serializable raw data>}
        """
        m21_stream =  safe_convert(music_xml_filenames[0], forceSource=True)
        m21_stream_with_metronome_marks = deepcopy(m21_stream)
        for i in range(1, len(music_xml_filenames)):
            next_m21_stream = safe_convert(music_xml_filenames[i], forceSource=True)
            next_m21_stream_with_metronome_marks = deepcopy(next_m21_stream)
            m21_stream.append(next_m21_stream)
            m21_stream_with_metronome_marks.append(next_m21_stream_with_metronome_marks)

        metronome_marks, time_signatures, first_measure_n = extract_metr_events_and_first_measure_number(m21_stream_with_metronome_marks)
        if len(metronome_marks) == 0:
            if metronome_mark is None:
                raise MissedMetronomeMarkException()
            metronome_marks = {first_measure_n : metronome_mark}
            original_bpm = False
        else:
            original_bpm = True
        m21_stream_with_metronome_marks = propagate_metr_events_to_parts(m21_stream_with_metronome_marks, metronome_marks)
        try:
            m21_expanded_stream_with_metronome_marks = m21_stream_with_metronome_marks.expandRepeats()
            cant_expand = False
        except:
            traceback.print_exc(file=sys.stdout)
            print("Can't expand score", sys.exc_info()[0])
            warnings_set.add("Can't expand score")
            cant_expand = True
            m21_expanded_stream_with_metronome_marks = m21_stream_with_metronome_marks

        result = self.extract_raw_data_map_from_streams(m21_stream, m21_stream_with_metronome_marks,
                                                      m21_expanded_stream_with_metronome_marks, warnings_set)
        result["original_bpm"] = original_bpm
        result["cant_expand"] = cant_expand
        return result


class BaseFeatureExtractor(FeaturesExtractor):
    """
    Extraction of the features which works similarly for all instruments.
    """
    def extract_raw_data_map_from_streams(
            self,
            m21_stream,
            m21_stream_with_metronome_marks,
            m21_expanded_stream_with_metronome_marks,
            warnings_set):
        raw_map = {}
        offsets, measure_offset_dict = get_measure_offsets(m21_stream)
        raw_map['offsets'] = offsets
        raw_map['measure_offset_dict'] = measure_offset_dict
        raw_map.update(rhythm_tempo_features(m21_expanded_stream_with_metronome_marks))
        unextended = rhythm_tempo_features(m21_stream_with_metronome_marks)
        raw_map["unextended_duration"] = unextended["duration"]

        raw_map.update(dynamics_general_features(m21_stream))

        return raw_map


class MonophonicFeatureExtractor(BaseFeatureExtractor):
    def extract_raw_data_map_from_streams(self, m21_stream, m21_stream_with_metronome_marks,
                                          m21_expanded_stream_with_metronome_marks, warnings_set):
        # TODO: here we assume, that analysed part is always first.
        # it should be specified/confirmed by the user before
        # score analysis.
        result = super().extract_raw_data_map_from_streams(
            m21_stream,
            m21_stream_with_metronome_marks,
            m21_expanded_stream_with_metronome_marks,
            warnings_set)
        part = m21_stream.getElementsByClass(stream.Part)[0]

        offset_map = build_offset_to_effective_bar_number_map(part)
        segments_by_bar, segments_by_meter_tempo = rhythm_segments(part, offset_map, warnings_set)

        result.update({
            "pitch_range" :pitch_range_per_part(part),
            "song_rhythm_info" : syncopation_statistics(
                part, offset_map, segments_by_bar, segments_by_meter_tempo, warnings_set).
                to_serializable_dict()
        })
        result.update(pitch_entropy([part]))
        result["avg_playing_speed"] = playing_speed_per_s_per_part(
            m21_stream_with_metronome_marks.getElementsByClass(stream.Part)[0])

        return result


class VocalFeatureExtractor(BaseFeatureExtractor):
    def extract_raw_data_map_from_streams(
            self,
            m21_stream,
            m21_stream_with_metronome_marks,
            m21_expanded_stream_with_metronome_marks,
            warnings_set):
        # TODO: vocal part is first, and sometimes contains errors in pickup bar duration
        # ("Here comes the sun"), so we take the first accompaniment part for bars counting.
        # Sort it out.

        result = super().extract_raw_data_map_from_streams(
            m21_stream,
            m21_stream_with_metronome_marks,
            m21_expanded_stream_with_metronome_marks,
            warnings_set)
        part = m21_stream.getElementsByClass(stream.Part)[0]

        offset_map = build_offset_to_effective_bar_number_map(part)
        segments_by_bar, segments_by_meter_tempo = rhythm_segments(part, offset_map, warnings_set)

        result.update({
            "pitch_range": pitch_range_per_part(part),
            "song_rhythm_info": syncopation_statistics(
                part, offset_map, segments_by_bar, segments_by_meter_tempo, warnings_set).to_serializable_dict()
        })
        result.update(pitch_entropy([part]))
        result["avg_playing_speed"] = playing_speed_per_s_per_part(
            m21_stream_with_metronome_marks.getElementsByClass(stream.Part)[0])

        return result


class KeysFeatureExtractor(BaseFeatureExtractor):
    def process_two_hands(self, lh_part, rh_part, key, f):
        return {'lh_' + key : f(lh_part), 'rh_' + key : f(rh_part)}

    def extract_rh_lh(self, m21_stream):
        # TODO: here we assume, that piano part is starting from the second.
        # The first one is vocal.
        # it should be specified/confirmed by the user before
        # score analysis.
        lh_part = m21_stream.getElementsByClass(stream.Part)[-1]
        rh_part = m21_stream.getElementsByClass(stream.Part)[-2]
        return rh_part, lh_part

    def extract_raw_data_map_from_streams(
            self,
            m21_stream,
            m21_stream_with_metronome_marks,
            m21_expanded_stream_with_metronome_marks,
            warnings_set):

        # Detect fingering.
        # compute_piece_argnn(m21_stream)

        rh_part, lh_part = self.extract_rh_lh(m21_stream)

        chord_density_dict = chord_density(m21_stream)

        result = super().extract_raw_data_map_from_streams(
            m21_stream, m21_stream_with_metronome_marks,
            m21_expanded_stream_with_metronome_marks, warnings_set)
        result.update(self.process_two_hands(
            lh_part, rh_part, 'pitch_range', pitch_range_per_part))

        result.update(self.process_two_hands(
            lh_part, rh_part, 'ornaments', ornaments_statistics))

        rh_events, lh_events, joint_events = performance_chain(rh_part, lh_part, set())
        rh_distribution = pitch_distributions(rh_events, result['bpm'])
        lh_distribution = pitch_distributions(lh_events, result['bpm'])

        result["rh_pitch_set_entropy"] = rh_distribution["pitch_set_entropy"]
        result["rh_pitch_class_set_entropy"] = rh_distribution["pitch_class_set_entropy"]
        result["rh_key_pattern_entropy"] = rh_distribution["key_pattern_entropy"]
        result["rh_pitch_set_entropy_rate"] = rh_distribution["pitch_set_entropy_rate"]
        result["rh_pitch_set_lz"] = rh_distribution["pitch_set_lz"]
        result["rh_pitch_class_set_entropy_rate"] = rh_distribution["pitch_class_set_entropy_rate"]
        result["rh_key_pattern_entropy_rate"] = rh_distribution["key_pattern_entropy_rate"]
        result["rh_pitch_intervals"] = rh_distribution["pitch_intervals"]
        result["rh_time_intervals"] = rh_distribution["time_intervals"]
        result["rh_pitch_sets"] = rh_distribution["pitch_sets"]
        result["rh_finger_sets"] = rh_distribution["finger_sets"]

        result["lh_pitch_set_entropy"] = lh_distribution["pitch_set_entropy"]
        result["lh_pitch_class_set_entropy"] = lh_distribution["pitch_class_set_entropy"]
        result["lh_key_pattern_entropy"] = lh_distribution["key_pattern_entropy"]
        result["lh_pitch_set_entropy_rate"] = lh_distribution["pitch_set_entropy_rate"]
        result["lh_pitch_set_lz"] = lh_distribution["pitch_set_lz"]
        result["lh_pitch_class_set_entropy_rate"] = lh_distribution["pitch_class_set_entropy_rate"]
        result["lh_key_pattern_entropy_rate"] = lh_distribution["key_pattern_entropy_rate"]
        result["lh_pitch_intervals"] = lh_distribution["pitch_intervals"]
        result["lh_time_intervals"] = lh_distribution["time_intervals"]
        result["lh_pitch_sets"] = lh_distribution["pitch_sets"]
        result["lh_finger_sets"] = lh_distribution["finger_sets"]

        result.update(pitch_distributions(joint_events, result['bpm']))
        result.update({
            # test feature
            # dynamics instrument specific
            "articulations": articulation_features(m21_stream),

            # Part Writing
            "avg_chord_density": chord_density_dict['avg_density'],
            "max_chord_density": chord_density_dict['max_density'],
            "avg_chord_spread": avg_chord_spread(m21_stream),
            "old_pitch_entropy": pitch_entropy(m21_stream.getElementsByClass(stream.Part)),

            # Rhythm Tempo
            "tuplets": tuplets_ratio(m21_stream)
        })

        # due to meter change in endings, only expanded stream works properly.
        #result.update({
        #    "lh_song_rhythm_info": syncopation_statistics(
        #        m21_expanded_stream_with_metronome_marks.getElementsByClass(stream.Part)[-1],
        #        warnings_set, decimation=False, ignore_tempo_marks=True).to_serializable_dict(),
        #    "rh_song_rhythm_info": syncopation_statistics(
        #        m21_expanded_stream_with_metronome_marks.getElementsByClass(stream.Part)[-2],
        #        warnings_set, decimation=False, ignore_tempo_marks=True).to_serializable_dict()
        #})

        rh_part, lh_part = self.extract_rh_lh(m21_stream_with_metronome_marks)

        offset_map = build_offset_to_effective_bar_number_map(m21_stream_with_metronome_marks)
        #lh_offset_map = offset_map
        # offset_map = build_offset_to_effective_bar_number_map(rh_part)
        # lh_offset_map = build_offset_to_effective_bar_number_map(lh_part)
        # who cares...
        #if offset_map != lh_offset_map:
        #    raise Exception("Offset/measure maps are not eqeal for left and right hands!")
        segments_by_bar, segments_by_meter_tempo = rhythm_segments(rh_part, offset_map, warnings_set, ignore_tempo_marks=True)
        lh_segments_by_bar, lh_segments_by_meter_tempo = rhythm_segments(lh_part, offset_map, warnings_set, ignore_tempo_marks=True)
        if len(lh_segments_by_meter_tempo) != len(lh_segments_by_meter_tempo):
            raise Exception("Different number of rhythm segments for left and right hands!")

        result.update(self.process_two_hands(
            lh_part, rh_part, 'clefs', lambda part, offsets = offset_map:clefs_statistics(part, offsets)))
        key_signature_map = key_signatures_statistics(rh_part, offset_map)
        key_signature_map.update(key_signatures_statistics(lh_part, offset_map))
        result["key_signatures"] = key_signature_map

        result.update({
            "lh_song_rhythm_info": syncopation_statistics(
                lh_part,
                offset_map,
                lh_segments_by_bar,
                lh_segments_by_meter_tempo,
                warnings_set, decimation=False).to_serializable_dict(),
            "rh_song_rhythm_info": syncopation_statistics(
                rh_part,
                offset_map,
                segments_by_bar,
                segments_by_meter_tempo,
                warnings_set, decimation=False).to_serializable_dict()
        })

        segments_by_bar, segments_by_meter_tempo = rhythm_segments(rh_part, offset_map, warnings_set, ignore_tempo_marks=True)
        lh_segments_by_bar, lh_segments_by_meter_tempo = rhythm_segments(lh_part, offset_map, warnings_set, ignore_tempo_marks=True)
        lh_stat, lh_list, lh_spans = syncopation_statistics_for_part(
                lh_part,
                offset_map,
                lh_segments_by_bar,
                lh_segments_by_meter_tempo,
                warnings_set, decimation=False)
        rh_stat, rh_list, rh_spans = syncopation_statistics_for_part(
                rh_part,
                offset_map,
                segments_by_bar,
                segments_by_meter_tempo,
                warnings_set, decimation=False)
        segments_by_bar, segments_by_meter_tempo = rhythm_segments(rh_part, offset_map, warnings_set, ignore_tempo_marks=True)
        joint_stat, _, _ = syncopation_statistics_for_lists(
                [rh_list, lh_list],
                offset_map,
                segments_by_bar,
                segments_by_meter_tempo,
                warnings_set, decimation=False)
        result.update({
            "new_lh_song_rhythm_info": lh_stat.to_serializable_dict(),
            "new_rh_song_rhythm_info": rh_stat.to_serializable_dict(),
            "joint_song_rhythm_info": joint_stat.to_serializable_dict(),
            "avg_playing_speed_per_score": 1000.0 / np.mean(np.concatenate((rh_spans, lh_spans)))
        })

        # TODO: "merged" syncopation. Currently, it seems hand parts are "glues" consequently.

        # Unstable (in terms of music21) and highly correlated with sum_*  features
        #s = stream.Score()
        #s.append(m21_expanded_stream_with_metronome_marks.getElementsByClass(stream.Part)[-2])
        #s.append(m21_expanded_stream_with_metronome_marks.getElementsByClass(stream.Part)[-1])
        #result["song_rhythm_info"] = syncopation_statistics(
        #    s, warnings_set, decimation=False).to_serializable_dict()

        return result


class CIPIFeatureExtractor(KeysFeatureExtractor):
    def extract_rh_lh(self, m21_stream):
        # TODO: here we assume, that piano part is starting from the second.
        # The first one is vocal.
        # it should be specified/confirmed by the user before
        # score analysis.
        lh_part = m21_stream.getElementsByClass(stream.Part)[1]
        rh_part = m21_stream.getElementsByClass(stream.Part)[0]
        return rh_part, lh_part


INSTRUMENT_FEATURE_EXTRACTORS = {
    'guitar':  MonophonicFeatureExtractor(),
    'vocal': VocalFeatureExtractor(),
    'keys': KeysFeatureExtractor(),
    'piano': KeysFeatureExtractor(),
    'cipi': CIPIFeatureExtractor(),
    'bass': MonophonicFeatureExtractor()
}


def get_args():
    """
    Parse arguments (if it's invoked as a command).
    """
    parser = argparse.ArgumentParser(
        description='Estimate score features raw data and store it to json file.')
    parser.add_argument(
        '-a', '--annotation', type=str, help='input annotation file in .xml file', required=True)
    parser.add_argument(
        '-o', '--output', type=str, default='output.json', help='output file in .json file (default output.json)',
        required=False)
    parser.add_argument(
        '-b', '--base', type=str, default='../..', help='base path (added to the score file paths from the annotation)',
        required=False)
    parser.add_argument(
        '-d', '--debug', action='store_true', default=False, help='Debug mode',
        required=False)
    parser.add_argument(
        '-f', '--files', type=str, help='Write individual files to the given directory',
        required=False)
    parser.add_argument(
        '-i', '--ignore-warnings', action='store_true', default=False, help='Ignore warnings, force to go on on in any situation',
        required=False)
    parser.add_argument(
        '-s', '--skip-written', action='store_true', default=True,
        help='In "individual files" mode, skip processing if target file already exists',
        required=False)
    parser.add_argument(
        '-c', '--correct-only', action='store_true', default=False, help='Consider only proofread entries',
        required=False)
    parser.add_argument(
        '-t', '--tempo',  type=int, default=None, help='tempo in bpm',
        required=False)
    parser.add_argument(
        '-p', '--bpm-file',  type=str, default=None, help='existing difficulties file with bpm',
        required=False)
    parser.add_argument(
        '-e', '--fatal-errors', type=str, default=None, help='File to output fata errors',
        required=False)
    args = parser.parse_args()
    infile = args.annotation
    outfile = args.output
    base_path = args.base
    debug = args.debug
    correct_only = args.correct_only
    files_directory = args.files
    ignore_warnings = args.ignore_warnings
    skip_written = args.skip_written
    tempo = args.tempo
    bpm_file = args.bpm_file
    errors_file = args.fatal_errors

    return infile, outfile, base_path, debug, correct_only, files_directory, ignore_warnings, skip_written, tempo, bpm_file, errors_file

##############################################

class DictAsArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        print("obj=", obj)
        if isinstance(obj, dict):
            return {"_dict_as_array" :[
                [json.JSONEncoder.default(self, i[0]), json.JSONEncoder.default(self, i[1])]
                for i in obj.items()]}
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def integer_keys_if_possible(dct):
    # analyse keys. if all are integers, convert to integer.
    all_ints = all([is_int(k) for k in dct.keys()])
    if all_ints:
        return {int(k):v for k,v in dct.items()}
    else:
        return dct


def dump_raw_data(data, fo):
    """
    Dumps raw data structure do json file.
    Parameters
    ----------
    data
    raw data object

    fo
    output stream
    """
    json.dump(data, fo, indent=True)


def load_raw_data(f):
    """
    Reads raw data structures from json file.

    Parameters
    ----------
    f input
    stream

    Returns
    -------
    Raw data dict
    """
    return json.load(f, object_hook=integer_keys_if_possible)


def print_warnings(warnings_set, file = sys.stdout):
    for w in warnings_set:
        print("WARNING: ", w, file = file)


def convert_offsets(data_map):
    data_map['offsets'] = [str(x) for x in data_map['offsets']]
    data_map['measure_offset_dict'] = {k:str(v) for k,v in data_map['measure_offset_dict'].items()}


def get_default_metronome_mark(id, default_metronome_mark, bpm_map):
    if bpm_map is None or id not in bpm_map:
        return default_metronome_mark
    else:
        return MetronomeMark(int(bpm_map[id]))


if __name__ == '__main__':
    in_file, out_file, base_path, debug, correct_only, files_directory, ignore_warnings, skip_written, tempo, bpm_file, errors_file = get_args()

    if tempo is None:
        default_metronome_mark = None
    else:
        default_metronome_mark = MetronomeMark(tempo)

    if bpm_file:
        with open(bpm_file) as f:
            content = json.load(f)
            bpm_map={x['id']:x['bpm'] for x in content}
    else:
        bpm_map = None

    songs = get_song_map(in_file, correct_only=correct_only).values()
    # determine instrument
    instruments = set([x['instrument'] for x in songs])
    if len(instruments) == 0:
        print("Instrument is undefined")
        exit(-1)
    elif len(instruments) > 1:
        print("Instrument is ambiguous", instruments)
        exit(-1)
    instrument = instruments.pop()

    features_extractor = INSTRUMENT_FEATURE_EXTRACTORS[instrument]

    res = []
    for s in songs:
        score_path = os.path.join(base_path, s['score'])
        print("Processing %s..." % score_path)
        if debug:
            print(score_path)
        if files_directory:
            path, name = os.path.split(score_path)
            name, ext = os.path.splitext(name)
            out_name = os.path.join(files_directory, os.path.extsep.join((name, "json")))
            if skip_written and exists(out_name):
                print("Skipping " + out_name)
                continue

        warnings_set = set()
        try:
            raw_data_map = features_extractor.extract_raw_data_map(
                [score_path],
                warnings_set,
                metronome_mark=get_default_metronome_mark(
                    s['id'], default_metronome_mark, bpm_map),
                ignore_warnings=ignore_warnings)
        except BaseException as err:
            if ignore_warnings:
                traceback.print_exc()
                warnings_set.add("Fatal error (nothing is written)")
                print_warnings(warnings_set)
                if errors_file is not None:
                    with open(errors_file, "a") as f:
                        print(s['id'], file = f)
                        traceback.print_exc(file=f)
                        print_warnings(warnings_set, file =f)
                        print("", f)
                continue
            else:
                raise err

        if len(warnings_set) > 0:
            print_warnings(warnings_set)
            if not ignore_warnings:
                raise Exception("Warnings found")

        raw_data_map['id'] = s['id']
        raw_data_map['grade'] = s['grade']

        convert_offsets(raw_data_map)

        if files_directory:
            with open(out_name, 'w') as fo:
                print("Writing %s..." % out_name)
                dump_raw_data(raw_data_map, fo)
        else:
            res.append(raw_data_map)

    if not files_directory:
        with open(out_file, 'w') as fo:
            print("Writing %s..." % out_file)
            dump_raw_data(res, fo)
    else:
        # merge all jsons from <files_directory>
        res = []
        json_files = glob.glob(os.path.join(files_directory, "*.json"))
        for jfile in json_files:
            with open(jfile, "rb") as f:
                res.append(load_raw_data(f))
        with open(out_file, 'w') as fo:
            print("Writing %s..." % out_file)
            dump_raw_data(res, fo)
