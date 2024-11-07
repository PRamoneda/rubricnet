import music21 as m21
import os
from copy import deepcopy
import numpy as np
from math import log2, modf
import pathlib
from statistics import mean
import pandas as pd
from common_utils import keepNotationOnly, recurseNotation, isNotatedChord, keepNotatedChordsOnly

part_aliases = {'LH': ['P2-Staff2', 'P1-Staff2', 'LH'], 'RH': ['P2-Staff1', 'P1-Staff1', 'RH']}
part_aliases_map = {'P2-Staff2': 'LH', 'P1-Staff2': 'LH', 'P2-Staff1': 'RH', 'P1-Staff1': 'RH'}


#utility function to return average pitch height of a chord
def avg_note(note):
    """ returns the avg midi value of a set of notes"""
    midi_list = [p.midi for p in note.pitches]
    return np.average(midi_list)


def articulation_features(m21_stream):
    '''
        TODO: Add slurs! apparently they are considered an articulation according to https://en.wikipedia.org/wiki/Articulation_(music)

        This function considers articulations that are indicated by dynamic marks (i.e rfz). However, it does not relate
        them to the regular accent articulation. They are treated as totally different.
        Inputs: music21 stream. Written for Keys so assumes that there are at least 2 'parts'
        Outputs: articulations rate and entropy
        Assumptions: stream has at least 2 parts, or it will crash 
                     Also, we need to account articulations that are represented in dynamic marks.
    '''
    articulational_dynamics = ['rfz', 'rf', 'fz', 'sfz', 'sf']
    articulation_frequencies = {}  # number of notes with each of the articulation values. none is a value as well
    num_notes = 0
    num_articulations = 0

    for part in m21_stream.parts[-2:]:
        measures = {}
        chords_and_notes = part.flat
        keepNotationOnly(chords_and_notes)

        dynamics = part.flat
        dynamics.removeByNotOfClass(['Dynamic'])

        for dynamic in dynamics:
            if dynamic.value in articulational_dynamics:
                measures[dynamic.measureNumber] = dynamic.value

        for note in chords_and_notes:
            marked = False  # flag to mark if a note shall be marked as articulated through one of function paths
            num_notes += 1
            # first, check articulation
            if note.articulations:
                marked = True
                for articulation in note.articulations:
                     articulation_frequencies[articulation.name] = articulation_frequencies.get(articulation, 0) + 1

            # then, check the scope of articulations that were indicated by dynamic marks
            if note.measureNumber in measures:
                marked = True
                dynamic_label = measures[note.measureNumber]
                articulation_frequencies[dynamic_label] = articulation_frequencies.get(dynamic_label, 0) + 1
            if not marked:
                articulation_frequencies['none'] = articulation_frequencies.get('none', 0) + 1

    # By here, we should have the articulation_frequencies ready.
    total_entropy = 0
    for articulation, frequency in articulation_frequencies.items():
        probability = float(frequency) / float(num_notes)
        log_probability = log2(probability)
        articulation_entropy = probability * log_probability
        total_entropy += articulation_entropy

    total_entropy = total_entropy * -1

    # calculate the total number of articulations for the articulations rate calculation:
    for key, value in articulation_frequencies.items():
        if key != 'none':
            num_articulations += value

    return {'articulations_entropy': total_entropy, 'articulation_frequencies': articulation_frequencies,
            'articulation_rate': 0 if num_notes == 0 else float(num_articulations)/float(num_notes)}


def tuplets_ratio(m21_stream):
    ''' Ratio of notes part of a tuplet to notes not part of a tuplet'''
    num_tuplets = 0
    tot_notes = 0

    for part in m21_stream.recurse().parts:
        if part.id not in part_aliases_map:
            continue
        part_alias = part_aliases_map[part.id]

        flat_part = part.flat
        keepNotationOnly(flat_part)

        for note in flat_part.notes:
            if note.duration.tuplets:
                num_tuplets += 1

        tot_notes += len(flat_part.notes)

    tuplets_ratio = 0 if tot_notes == 0 else float(num_tuplets)/float(tot_notes)

    return tuplets_ratio


def difficulty_per_offset(m21_stream):
    # Apparently it doesn't give difficulty per second.. it gives prob difficulty score by note! which is interesting..
    # to covert it to something easy to display: group the values (by taking average) per offset unit. take the bh score
    # There is a bug that could arise if the ordering of notes is non uniform. In the difficulty calc files, the chord
    # notes are shown from top to bottom. In music21, it is from bottom to top (like the typical reading of it..)

    dir_path = os.path.dirname(os.path.realpath(__file__))  # since the difficulty calculations are assumed to be in same dir
    difficulty_directory = 'RPSecondsDifficulty'
    path = pathlib.PurePath(m21_stream.filePath)
    per_second_file_path = os.path.join(dir_path, difficulty_directory, '{}.tsv'.format(path.name[:-4]))
    summary_file_path = os.path.join(dir_path, "rp_difficulty_with_grades.csv")

    difficulty_per_note_df = pd.read_csv(per_second_file_path, sep='\t')
    difficulty_summary_df = pd.read_csv(summary_file_path)

    difficulty_df = difficulty_per_note_df[["spelled pitch", "diffBH"]]

    lh = recurseNotation(m21_stream.parts[-1].stripTies()).offsetMap()
    rh = recurseNotation(m21_stream.parts[-2].stripTies()).offsetMap()

    #remove voice, create offset map of both hands together. put

    difficulty_per_offset_unit = {}  # only per offset whole number..
    # load the pitch and diffBH columns..
    difficulty_average = {}

    m21_stream = deepcopy(m21_stream)  # overwriting the reference with a copy..

    top_lh = 0
    top_rh = 0

    processed_indexes = []

    for index, row in difficulty_df.iterrows():
        # is this pitch in lh?
        if index in processed_indexes:  # since we skip the normal loop in the chord processing, we had to do this..
            continue
        print('processing {}'.format(row['spelled pitch']))

        processed_indexes.append(index)

        if top_lh < len(lh) and m21.pitch.Pitch(row['spelled pitch']).ps \
                in [p.ps for p in lh[top_lh].element.pitches]:

            # if chord: process full chord and advance top
            if isNotatedChord(lh[top_lh].element):
                curr_index = index
                chord_from_score = m21.chord.Chord()
                chord_from_diff_file = m21.chord.Chord()
                for pitch in lh[top_lh].element.pitches:
                    chord_from_score.add(int(pitch.ps))
                    chord_from_diff_file.add(difficulty_df.at[curr_index, 'spelled pitch'])
                    processed_indexes.append(curr_index)
                    curr_index += 1

                if chord_from_score.sortChromaticAscending().pitchNames != \
                        chord_from_diff_file.sortChromaticAscending().pitchNames:
                    print('Chords Not Equal!! Problem!!')

            whole_offset = int(lh[top_lh].offset)
            print('lh offset {}, Measure: {}, row_in_df: {}, pitch: {}'.format(
                whole_offset,
                lh[top_lh].element.measureNumber, index,
                'from_score {} - from_diff_file {}'.format(chord_from_score, chord_from_diff_file)
                if isNotatedChord(lh[top_lh].element) else row['spelled pitch']))

            if whole_offset not in difficulty_per_offset_unit:
                difficulty_per_offset_unit[whole_offset] = []
            difficulty_per_offset_unit[whole_offset].append(row['diffBH'])

            # advance:
            top_lh += 1

        # is this pitch in rh?
        elif top_rh < len(rh) and m21.pitch.Pitch(row['spelled pitch']).ps \
                in [p.ps for p in rh[top_rh].element.pitches]:

            # if chord: process full chord and advance top
            if isNotatedChord(rh[top_rh].element):
                curr_index = index
                chord_from_score = m21.chord.Chord()
                chord_from_diff_file = m21.chord.Chord()
                for pitch in rh[top_rh].element.pitches:
                    chord_from_score.add(int(pitch.ps))
                    chord_from_diff_file.add(difficulty_df.at[curr_index, 'spelled pitch'])
                    processed_indexes.append(curr_index)
                    curr_index += 1

                if chord_from_score.sortChromaticAscending().pitchNames != \
                        chord_from_diff_file.sortChromaticAscending().pitchNames:
                    print('Chords Not Equal!! Problem!!')

            whole_offset = int(rh[top_rh].offset)
            print('rh offset {}, Measure: {}, row_in_df: {}, pitch: {}'.format(
                whole_offset,
                rh[top_rh].element.measureNumber, index,
                'from_score {} - from_diff_file {}'.format(chord_from_score, chord_from_diff_file)
                if isNotatedChord(rh[top_rh].element) else row['spelled pitch']))

            if whole_offset not in difficulty_per_offset_unit:
                difficulty_per_offset_unit[whole_offset] = []
            difficulty_per_offset_unit[whole_offset].append(row['diffBH'])

            # advance:
            top_rh += 1

        else:
            print("error. stopping the loop. index:{}, pitch:{}, rh_top:{}, lh_top: {}".format(
                index, row['spelled pitch'], rh[top_rh], lh[top_lh]))
            return ''

    # average the difficulties per each whole offset number
    for key, value in difficulty_per_offset_unit.items():
        difficulty_per_offset_unit[key] = mean(value)

    return {'difficulty_per_offset': difficulty_per_offset_unit, 'difficulty_average':
        difficulty_summary_df.loc[difficulty_summary_df['fname'] == '{}.mid'.format(path.name[:-4])]['diffBH'].values[
            0]}


def chord_density(m21_stream):

    num_chords = 0
    sum_density = 0
    max_density = 1

    for part in m21_stream.getElementsByClass(m21.stream.Part)[-2:]:
        flat_part = part.flat
        keepNotationOnly(flat_part)

        for note in flat_part.notes:
            if isNotatedChord(note):
                num_chords += 1
                sum_density += len(note.pitches)
                if len(note.pitches) > max_density:
                    max_density = len(note.pitches)

    chord_density = 0 if num_chords == 0 else float(sum_density)/float(num_chords)

    return {'avg_density': chord_density, 'max_density': max_density}


def avg_chord_spread(m21_stream):
    sum_diff = 0
    num_chords = 0

    for part in m21_stream.getElementsByClass(m21.stream.Part)[-2:]:
        flat_score = part.flat
        keepNotatedChordsOnly(flat_score)
        num_chords += len(flat_score.notes)

        for chord in flat_score.notes:
            # The array comes sorted..
            sum_diff += chord.pitches[-1].ps - chord.pitches[0].ps

    avg_spread = 0 if num_chords == 0 else float(sum_diff) / num_chords

    return avg_spread



