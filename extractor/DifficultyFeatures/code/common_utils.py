import copy

from lxml import etree
from music21 import converter, meter, tempo, stream
import sys
import traceback
import time
import re
import glob
import music21 as m21
from music21.stream.filters import ClassNotFilter
from collections import Counter

from math import log2, modf
import numpy as np
import math
from collections import namedtuple
# TODO: common logging approach.


def isNotatedChord(obj):
    """

    Parameters
    ----------
    obj - music21 object

    Returns true if it's a notated chord (not a Harmony symbol).
    -------

    """
    return isinstance(obj, m21.chord.Chord) and not isinstance(obj, m21.harmony.Harmony)

def keepNotationOnly(stream):
    """
    Performs stream filtering: keeps only notation
    (notes and common chords, but not Harmony symbols)
    Parameters
    ----------
    stream

    Returns nothing
    -------

    """
    stream.removeByNotOfClass([m21.note.Note, m21.chord.Chord])
    stream.removeByClass([m21.harmony.Harmony])


def keepNotatedChordsOnly(stream):
    """
    Performs stream filtering: keeps only notated chords (but not Harmony symbols)
    Parameters
    ----------
    stream

    Returns nothing
    -------

    """
    stream.removeByNotOfClass([m21.chord.Chord])
    stream.removeByClass([m21.harmony.Harmony])


def recurseNotation(stream):
    return stream.recurse(classFilter=['Note', 'Chord']).addFilter(ClassNotFilter(['Harmony']))


def recurseNotationAndRests(stream):
    return stream.recurse(classFilter=['Note', 'Chord', 'Rest']).addFilter(ClassNotFilter(['Harmony']))


def get_measure_offsets(m21_stream):
    """ Function to help with displaying local feature values"""
    offsets = m21_stream.measureOffsetMap()
    measure_offset_dict = {}
    for offset, element in offsets.items():
        measure_offset_dict[element[0].number] = offset  # don't know if specifically indexing 0 will generalize well
    return list(offsets.keys()), measure_offset_dict


def pitch_color_map_gen(pitch_array):
    #todo: change this to be dependent on pitch midi numbers to avoid problems that could occur with double sharpes
    #or double flats. Also the sorting wouldn't need any special code..

    pitch_dict1 = {"B#": "#f7d148","C": "#f7d148", "G": "#81d94a", "D": "#278f4b", "A": "#278791", "E": "#276a91", "B": "#305b9c",
                   "F#": "#282f7d", "C#": "#99268c", "G#": "#c9182d", "D#": "#c93818", "A#": "#ee6216", "E#": "#e08026",
                   "F": "#e08026"}
    pitch_dict2 = {"C": "#f7d148", "G": "#81d94a", "D": "#278f4b", "A": "#278791", "G##": "#278791", "E": "#276a91", "C-": "#305b9c",
                   "G-": "#282f7d", "D-": "#99268c", "A-": "#c9182d", "E-": "#c93818", "B-": "#ee6216", "C--": "#ee6216", "F": "#e08026",
                   "F-": "#276a91", "A--": "#81d94a", "B--": "#278791", "D--": "#f7d148", "F--": "#c93818", "F##":  "#81d94a"}

    pitch_ordering = {"C": 1, "B#": 1, "D--": 1, "A--": 2, "G": 2, "F##": 2, "D": 3, "A": 4, "G##": 4, "B--": 4, "E": 5, "F-": 5, "B": 6, "F#": 7, "C#": 8, "G#": 9, "D#": 10,
                      "A#": 11, "E#": 12, "F": 12, "C-": 6, "G-": 7, "D-": 8, "A-": 9, "E-": 10, "F--": 10, "B-": 11, "C--": 11}

    # sort, so the colors display looks ok.
    sorted_pitches = sorted(list(pitch_array.keys()), key=lambda x: pitch_ordering[x[0:-1]])

    # generate sorted freq array and colors array
    frequencies = []
    colors = []
    for pitch in sorted_pitches:
        frequencies.append(pitch_array[pitch])
        if pitch[0:-1] in pitch_dict1:
            colors.append(pitch_dict1[pitch[0:-1]])
        else:
            colors.append(pitch_dict2[pitch[0:-1]])

    return sorted_pitches, frequencies, colors

# ------ Playing Speed Feature Start -------

#def playing_speed_local_range(speed_dict, offsets, measure_offset_dict):
#    changes = {x: 0 for x in offsets}
#
#    for measure_num, speed in speed_dict.items():
#        offset = measure_offset_dict[measure_num]
#        changes[offset] = speed
#
#    # change the measure no to offset no
#    return list(changes.values())


def extract_metr_events_and_first_measure_number(m21_stream):
    """
    Extracts MetronomeMarks from the stream.

    Parameters
    ----------
    m21_stream - streram to process

    Returns
    -------
    Dictionary measure_number -> m21.tempo.MetronomeMark
    """
    metronome_marks = {}
    time_signatures = {}
    for mm in m21_stream.recurse().getElementsByClass(m21.tempo.MetronomeMark):
        measure = find_parent(mm, m21.stream.Measure)
        metronome_marks[measure.number] = mm
    for mm in m21_stream.recurse().getElementsByClass(m21.meter.TimeSignature):
        measure = find_parent(mm, m21.stream.Measure)
        time_signatures[measure.number] = mm
    m_number = None
    for m in m21_stream.recurse(classFilter='Measure'):
        if m_number is None or m_number > int(m.number):
            m_number = int(m.number)
    return metronome_marks, time_signatures, m_number


def propagate_metr_events_to_parts(m21_stream, metronome_marks):
    """
    Inserts the Metronome Marks in every part (if it's missed)
    so that .seconds attribute can work correctly

    Parameters
    ----------
    m21_stream - stream to proecess
    metronome_marks -  Dictionary: measure_number -> m21.tempo.MetronomeMark

    Returns
    -------
    Music21 stream with the marks inserted.

    """
    for measure_number, mm in metronome_marks.items():
        measure_stream = m21_stream.measures(measure_number, measure_number)
        for measure in measure_stream.recurse().getElementsByClass(m21.stream.Measure):
            existing = measure.getElementsByClass(type(mm))
            if len(existing) == 0:
                # TODO: deep copy?
                measure.insert(0, mm)
    # for measure_number, mm in time_signatures.items():
    #    measure_stream = m21_stream.measures(measure_number, measure_number)
    #    for measure in measure_stream.recurse().getElementsByClass(m21.stream.Measure):
    #        existing = measure.getElementsByClass(type(mm))
    #        if len(existing) == 0:
    #            measure.insert(0, mm)
    return m21_stream


def playing_speed_per_s_per_part(part_with_metronome_marks):
    """

    Parameters
    ----------
    part_with_metronome_marks - part which has metronome marks (e.g., if
    the part is not the top one, metranome marks are likely to be propagated
    from the top part, e.g. with propagate_metronome_mark_to_parts function.

    Returns
    -------
    A dictionary with two entries:
    -  mapping from measures to avg speed per measure
    - value for avg speed

    The 'playing speed' at each measure as defined by the Chui Paper.
    It can be also seen as the average
    seconds taken by each note in a measure
    Updated from master thesis version to have a playing speed per part

    """
    total_speed = 0
    total_notes = 0

    """
    Expands ties before calculation. This might introduce a problem that
    some measures will be have duration parts 
    for things they are tied to in other measures
    """
    #part = part.stripTies() #think about whether or not it makes sense to have stripped ties here..
    avg_playing_speed_per_measure = {}

    for measure in part_with_metronome_marks.getElementsByClass(m21.stream.Measure):
        try:
            seconds = [note.seconds for note in measure.notes]
            if seconds:
                playing_speed = np.average(seconds)
            else:
                playing_speed = 0

            total_speed += sum([x for x in seconds if not math.isnan(x)])
            total_notes += len(seconds)

            avg_playing_speed_per_measure.update({measure.number: 0 if playing_speed == 0 or math.isnan(playing_speed)
                                                            else float(1.0/playing_speed)})

        # post process the
        except m21.exceptions21.Music21Exception:
            print('{} - {}'.format(measure, measure.activeSite))  # Monitor when this bug happens..
            print('metronome mark missing, skipping measure')

    avg_playing_speed_per_score = 0 if total_notes == 0 else float(total_notes) / float(total_speed)

    return {'avg_playing_speed_per_measure': avg_playing_speed_per_measure,
            'avg_playing_speed_per_score': avg_playing_speed_per_score}


# ------ Playing Speed Feature End -------


class Stopwatch:
    def __init__(self, caption, debug = False):
        self.caption = caption
        self.debug = debug
        if self.debug:
            print("[Stopwatch] %s..." % (self.caption))

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.delta = self.end_time - self.start_time
        if self.debug:
            print("[Stopwatch] %s: %f" % (self.caption, self.end_time - self.start_time))


def safe_convert(score_file_path, forceSource=False):
    try:
        s = converter.parse(score_file_path, forceSource=forceSource)
    except:
        traceback.print_exc(file=sys.stdout)
        print("Can't convert", score_file_path, sys.exc_info()[0])
        s = None
    return s


def find_parent(el, class_type):
    """
    Returns the first parent of el of type class_type. Both el and class_type must be valid music21 objects
    If class_type does not exist in the parent chain of el, the outermost object will be returned
    """
    temp = el
    while not isinstance(temp, class_type) and temp.activeSite:
        temp = temp.activeSite
    return temp


def find_time_signature(measures):
    for m in measures:
        t_s = m.getElementsByClass(meter.TimeSignature).stream()
        if len(t_s) > 0:
            return t_s[0]
    raise Exception('No time signature')


def find_metronome_mark(measures):
    for m in measures:
        for mm in m.getElementsByClass(tempo.MetronomeMark).stream():
            if not mm.number is None:
                return mm
    for m in measures:
        for mm in m.getElementsByClass(tempo.MetronomeMark).stream():
            if not mm.numberSounding is None:
                return mm

    raise Exception('No metronome mark')


def rhythm_tempo_features(expanded_score):
    expandedParts = expanded_score.parts.stream()
    top_part = expandedParts[0]
    measures = top_part.getElementsByClass('Measure').stream()
    # TODO: questionable approach. time_signature and metrnome mark
    # should be defined from the very beginning.
    time_signature = find_time_signature(measures)
    metronomeMark = find_metronome_mark(measures)
    bpm_ref = metronomeMark.referent.type
    if not metronomeMark.number is None:
        bpm_orig = metronomeMark.number
        bpm = metronomeMark.getEquivalentByReferent(1.0).number
    else:
        effectiveMark = m21.tempo.MetronomeMark(number=metronomeMark.numberSounding, referent=metronomeMark.referent)
        bpm_orig = effectiveMark.number
        bpm = effectiveMark.getEquivalentByReferent(1.0).number
    duration_time = metronomeMark.durationToSeconds(
        expandedParts.duration)

    return {
        'time_signature': time_signature.ratioString,
        'bpm': bpm,
        'bpm_orig': bpm_orig,
        'bpm_ref': bpm_ref,
        'duration':duration_time
    }


# ------- Pitch Range Features Start ---------

def pitch_range_per_part(m21_part):
    """ gets the max pitch value and min pitch value corresponding to the music 21 part for the whole score"""
    max_pitch = 0
    min_pitch = 200

    for note in m21_part.flat.notes:
        if isinstance(note, m21.note.Note) or isNotatedChord(note):
            for pitch in note.pitches:
                max_pitch = pitch.ps if pitch.ps > max_pitch else max_pitch
                min_pitch = pitch.ps if pitch.ps < min_pitch else min_pitch

    return (min_pitch, max_pitch)


# -------- Pitch Range Features End ---------


def pitch_entropy(m21_parts):
    """ pitch entropy as defined by the Chiu Paper. Perhaps it is a better rep. than accidentals
        change. However, it cannot be calculated per measure and then grouped easily, so perhaps accidentals
        change can be kept as a way to visualize the per measure change, and pitch entropy can be
        more of the summative one. But, since we want to cluster based on different granularities,
        then we must calculated on a measure level and then group them. But perhaps it doesn't make
        too much sense to maintain on a measure level. Maybe phrase level, but measure level doesn't make
        too much sense."""

    # calculate all the pitches in the score (except the vocal part).
    # map of keys (pitch number in midi) and occurrences.

    all_notes = []
    for p in m21_parts:
        all_notes.extend(p.flat.notes)

    pitch_frequency_map = {}
    all_frequencies = 0

    for note in all_notes:
        if isinstance(note, m21.note.Note):
            pitch_frequency_map[note.nameWithOctave] = pitch_frequency_map.get(note.nameWithOctave, 0) + 1
            all_frequencies = all_frequencies + 1
        elif isNotatedChord(note):
            for pitch in note.pitches:
                pitch_frequency_map[pitch.nameWithOctave] = pitch_frequency_map.get(pitch.nameWithOctave, 0) + 1
                all_frequencies = all_frequencies + 1

    total_entropy = 0

    for note, frequency in pitch_frequency_map.items():
        probability = float(frequency) / float(all_frequencies)
        #print('{} probability in {}: {}'.format(note, self.song_title, probability))
        log_probability = log2(probability)
        pitch_entropy = probability * log_probability
        total_entropy += pitch_entropy

    total_entropy = total_entropy * -1

    return {'pitch_entropy': total_entropy, 'pitch_frequency_map': pitch_frequency_map}


# -------- Dynamics General Features Start -------------


def dynamics_change_local_range(tuples, offsets, measure_offset_dict):
    """ The local value for dynamic change returns an array of tuples, where each tuple is:
    (measure of previous dynamic, measure of current dynamic, magnitude and direction of the change). This is useful
    for display.
    Inputs:
    ------
        tuples: array of aforementioned tuples representing dynamic changes in a song
        num_measures: number of measures of the song
    Outputs:
    -------
        returns array with the dynamic change value per measure. (Assumes one change per measure for now to facilitate
        the sampling """

    changes = {x: 0 for x in offsets}

    for from_measure, to_measure, value in tuples:
        offset = measure_offset_dict[to_measure]
        changes[offset] = changes.get(offset, 0) + abs(value)  # not sure to use abs or to change it

    #change the measure no to offset no
    return list(changes.values())


def dynamics_change_by_measure(tuples, measures):
    """ The local value for dynamic change returns an array of tuples, where each tuple is:
    (measure of previous dynamic, measure of current dynamic, magnitude and direction of the change). This is useful
    for display.
    Inputs:
    ------
        tuples: array of aforementioned tuples representing dynamic changes in a song
        num_measures: number of measures of the song
    Outputs:
    -------
        returns array with the dynamic change value per measure. (Assumes one change per measure for now to facilitate
        the sampling """

    changes = {x: 0 for x in measures}

    for from_measure, to_measure, value in tuples:
        changes[to_measure] = changes.get(to_measure, 0) + abs(value)  # not sure to use abs or to change it

    return changes


'''
def dynamics_change_global_range(plain_value_list):
    global_vals = []
    for plain_val in plain_value_list:
        global_vals.append(plain_val['dynamics_change_rate'])
    return global_vals
'''


def dynamics_change(m21_stream):
    '''
        TODO:
        - Take into account suddenness of crescendo and diminuendo marks.
        - ppp coupled with dense chords is harder than ppp in the general sense. This needs to be accounted for
        somehow. Features in relation to others from differing dimensions.
        ------
        This feature boldly assumes that a dynamic itself does not contribute to difficulty,
        but more so the changes in dynamics. A dynamic change can be summarized as a magnitude of direction and diff.
        Note: Articulations indicated
        by dynamic marks are taken into account since they tend to imply a dynamic as well, but the suddenness of the
        change does not contribute to the score. Handles fp as if it were 2 close f then p dynamics.
        Inputs: m21_stream
        Outputs: dynamics change: array of tuples where each is (from measure, to measure, dynamic change),
                 and dynamics change rate: which is the sum of contrast magnitudes / number of changes
    '''
    # TODO: does it make semse to consider sforzando and "other dynamics" in the same list as pinao/forte?
    values = {'sffz': 7, 'rfz': 6, 'rf': 6, 'fz': 6, 'sfz': 6, 'sf': 6, 'fffff': 10, 'ffff': 9, 'fff': 8, 'ff': 7, 'f': 6, 'mf': 5, 'other-dynamics':4.5, 'mp': 4,
             'p': 3, 'pp': 2, 'ppp': 1, 'pppp':0, 'ppppp':-1, 'sfp':3, 'sfpp':2}
    contrasted = {'fp': ['f', 'p']}  # to handle the dynamic changes that are in themselves a rep. of contrast

    num_changes = 0
    contrast_sum = 0

    dynamics_change_result = []

    dynamics = m21_stream.recurse().getElementsByClass(m21.dynamics.Dynamic)

    for dc1, dc2 in zip(dynamics[:-1], dynamics[1:]):
        dc1_val = dc1.value
        dc2_val = dc2.value

        if dc1_val in contrasted:
            # create a dc from first to second contrasted item
            temp_dc1 = contrasted[dc1_val][0]
            temp_dc2 = contrasted[dc1_val][1]

            difference = values[temp_dc1] - values[temp_dc2]
            dynamics_change_result.append((dc1.activeSite.number, dc1.activeSite.number, difference))
            num_changes += 1
            contrast_sum += abs(difference)

            # set dc1 to the 'to' dynamic
            dc1_val = temp_dc2

        if dc2_val in contrasted:
            # just set dc2 to the 'from'. the contrast itself will be captured in next iter.
            dc2_val = contrasted[dc2_val][0]

        difference = values[dc2_val] - values[dc1_val]
        if abs(difference) > 0:
            dynamics_change_result.append((dc1.activeSite.number, dc2.activeSite.number, difference))
            num_changes += 1
            contrast_sum += abs(difference)

    return {'dynamics_change': dynamics_change_result, 'dynamics_change_rate': 0 if num_changes == 0
            else float(contrast_sum)/float(num_changes) }


def dynamics_general_features(score):
    num_measures = len(score.recurse().getElementsByClass(stream.Measure))
    result = dynamics_change(score)
    result['num_measures'] = num_measures
    return result
# --------- Dynamics General Features End --------------


def bars_number_feature(score_part):
    """
    Provide test data for "score local" features visualization.

    Parameters
    ----------
    score_part: music21 score part.
    For vocal, it's recommended to use accomaniment part (N1),
    because vocal part could contains wrong pickup bar duration.

    Returns number of measures.
    -------

    """
    return len(score_part.getElementsByClass(stream.Measure))


def get_song_map(db_xml, correct_only=False):
    res = {}
    with open(db_xml, "rb") as f:
        tree = etree.parse(f)
        data = tree.getroot()
        for s in data.iter("song"):
            score = s.find("score")
            if not correct_only or \
                (score.get("proofread") and score.get("proofread").lower() == "true"):

                mbidable = s.find("mbidable")
                if mbidable is not None:
                    mbidable = mbidable.text
                else:
                    mbidable = ''

                meta_description = s.find("meta-description")
                if meta_description is not None:
                    meta_description = meta_description.text
                else:
                    meta_description = ''

                performance_tips = s.find("performance-tips")
                if performance_tips is not None:
                    performance_tips = performance_tips.text
                else:
                    performance_tips = ''

                demo = s.find("audio[@kind='demo']")
                if demo is not None:
                    demo = demo.get("path")
                else:
                    demo = ''

                minus_one = s.find("audio[@kind='minus-one']")
                if minus_one is not None:
                    minus_one = minus_one.get("path")
                else:
                    minus_one = ''

                new_one = {
                    "id":s.get("id"),
                    "technical-focus":bool(s.get("technical-focus")),
                    "instrument":s.get("instrument"),
                    "grade":int(s.get("grade")),
                    "mbidable":mbidable,
                    "meta-description": meta_description,
                    "performance-tips": performance_tips,
                    "score": score.get("path"),
                    "demo": demo,
                    "minus-one": minus_one,
                    "proofread": score.get("proofread", True)
                }
                mbids = []
                if s.find("mbids"):
                    for m in s.find("mbids").iter("mbid"):
                        mbids.append(m)
                    new_one["mbids"] = mbids
                new_one.update(s.items())
                res[new_one["id"]] = new_one
    return res


def get_songs_vocal(db_xml):
    res = []
    with open(db_xml, "rb") as f:
        tree = etree.parse(f)
        data = tree.getroot()
        for s in data.iter("song"):
            new_one = {
                "id":s.get("id"),
                "technical-focus":bool(s.get("technical-focus")),
                "instrument":s.get("instrument"),
                "grade":int(s.get("grade")),
                "mbidable":s.find("mbidable").text,
                "meta-description": s.find("meta-description").text,
                "performance-tips": s.find("performance-tips").text,
                "score": s.find("score").get("path"),
                "demo": s.find("audio[@kind='demo']").get("path"),
                "minus-one": s.find("audio[@kind='minus-one']").get("path"),
                "stem-vocal-source": s.find("audio[@kind='stem-vocal']").get("path")
            }
            mbids = []
            for m in s.find("mbids").iter("mbid"):
                mbids.append(s.items())
            new_one["mbids"] = mbids
            new_one.update(s.items())
            res.append(new_one)
    return res



def build_offset_to_effective_bar_number_map(m21part):
    # remap to effective bar numbers
    # music21 measure.number is not unique in case of several endings (volta brackets),
    # and in case of repeat expansion.
    #
    offset_map = m21part.measureOffsetMap()
    # mimic music21 approach. Normally it's 1, but in case of pick up bar, it's 0.
    effective_bar_number = m21part.recurse(classFilter=m21.stream.Measure)[0].number
    offset_to_effective_number = {}
    for k in offset_map.keys():
        offset_to_effective_number[k] = effective_bar_number
        effective_bar_number += 1

    return offset_to_effective_number

#lz_complexity("0001101001000101")
#0*001*10*100*1000*101

def lz_complexity(s):
    p = 0
    C = 1
    #  length of the current prefix
    u = 1
    #  length of the current component for the current pointer p
    v = 1
    vmax = v
    #print(s[0:1], end='|')
    while u + v < len(s):
        if s[p - 1 + v] == s[u + v - 1]:
            v += 1
        else:
            vmax = max(v, vmax)
            p += 1
            if p == u:
                #print(s[p:u+vmax], end='|')
                C += 1
                u += vmax
                v = 1
                p = 0
                vmax = v
            else:
                v = 1
    if v > 1:
        C += 1
        #print(s[u:])
    return C


def lz_complexity_norm(s):
    p = len(set(s))
    if p <2:
        p = 2
    n = len(s)
    if n < 2:
        n = 2
    return lz_complexity(s) / ( n / math.log(n, p))


def ornaments_statistics(m21part):
    expressions = []
    for note in m21part.flat.notes:
        if (isinstance(note, m21.note.Note) or isNotatedChord(note)):
            if note.expressions:
                for expr in note.expressions:
                    if expr.__class__.__module__ == 'music21.expressions':
                        expressions.append(expr.__class__.__name__)
            expressions.append("note")
    return dict(Counter(expressions))