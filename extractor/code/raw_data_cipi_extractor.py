import argparse
from music21.tempo import MetronomeMark
import os
import sys
from os.path import exists
import traceback
import glob
import json

from tqdm import tqdm

from raw_data_extractors import INSTRUMENT_FEATURE_EXTRACTORS,\
    print_warnings, get_default_metronome_mark, convert_offsets, dump_raw_data, load_raw_data

def get_args():
    """
    Parse arguments (if it's invoked as a command).
    """
    parser = argparse.ArgumentParser(
        description='Estimate score features raw data and store it to json file.')
    parser.add_argument(
        '-a', '--annotation', type=str, help='input annotation file in .json file', required=True)
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
        '-t', '--tempo',  type=int, default=None, help='tempo in bpm',
        required=False)
    parser.add_argument(
        '-e', '--fatal-errors', type=str, default=None, help='File to output fata errors',
        required=False)
    args = parser.parse_args()
    infile = args.annotation
    outfile = args.output
    base_path = args.base
    debug = args.debug
    files_directory = args.files
    ignore_warnings = args.ignore_warnings
    skip_written = args.skip_written
    tempo = args.tempo
    errors_file = args.fatal_errors

    return infile, outfile, base_path, debug, files_directory, ignore_warnings, skip_written, tempo, errors_file


def get_songs(in_file):
    # scores
    # id
    # instrument = cipi
    # grade
    # name
    # composer
    with open(in_file) as inf:
        index_data = json.load(inf)
    res = []
    for k, v in index_data.items():
        res.append({
            "id": k,
            "name": v["work_name"],
            "composer":  v["composer"],
            "grade": v["henle"],
            "instrument": "cipi",
            "scores": list(v["path"].values())
        })
    return res

##############################################

if __name__ == '__main__':
    in_file, out_file, base_path, debug, files_directory, ignore_warnings, skip_written, tempo, errors_file = get_args()

    if tempo is None:
        default_metronome_mark = None
    else:
        default_metronome_mark = MetronomeMark(tempo)

    bpm_map = None

    songs = get_songs(in_file)
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
    for s in tqdm(songs):
        score_paths = [os.path.join(base_path, x) for x in s['scores']]
        for score_path in score_paths:
            if debug:
                print("Processing %s..." % score_path)
            # tree = etree.parse(score_path)
            # for n in tree.iter("note"):
            #     if n.find("cue") is not None and n.find("tie") is not None:
            #         print(s['id'], " cue and tie !!!!")
            #         if errors_file is not None:
            #             with open(errors_file, "a") as f:
            #                 print(s['id'], " cue and tie !!!!", file=f)

        if files_directory:
            out_name = os.path.join(files_directory, os.path.extsep.join((s['id'], "json")))
            if skip_written and exists(out_name):
                print("Skipping " + out_name)
                continue

        warnings_set = set()
        try:
            raw_data_map = features_extractor.extract_raw_data_map(
                score_paths,
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
