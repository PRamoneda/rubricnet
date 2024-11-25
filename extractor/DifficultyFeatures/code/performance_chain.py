from music21 import stream, pitch
import music21 as m21
from fractions import Fraction
from extractor.DifficultyFeatures.code.common_utils import recurseNotationAndRests, find_parent, lz_complexity
from scipy.stats import entropy
from collections import Counter
import numpy as np

from extractor.DifficultyFeatures.code.piano_fingering_argnn.compute_embeddings import compute_piece_argnn

MIDI_TO_NEAREST_WHITES = {
    0: (0, 0),
    1: (0, 1),
    2: (1, 1),
    3: (1, 2),
    4: (2, 2),
    5: (3, 3),
    6: (3, 4),
    7: (4, 4),
    8: (4, 5),
    9: (5, 5),
    10: (5, 6),
    11: (6, 6),
    12: (7, 7),
    13: (7, 8),
    14: (8, 8),
    15: (8, 9),
    16: (9, 9),
    17: (10, 10),
    18: (10, 11),
    19: (11, 11),
    20: (11, 12),
    21: (12, 12),
    22: (12, 13),
    23: (13, 13),
    24: (14, 14),
    25: (14, 15),
    26: (15, 15),
    27: (15, 16),
    28: (16, 16),
    29: (17, 17),
    30: (17, 18),
    31: (18, 18),
    32: (18, 19),
    33: (19, 19),
    34: (19, 20),
    35: (20, 20),
    36: (21, 21),
    37: (21, 22),
    38: (22, 22),
    39: (22, 23),
    40: (23, 23),
    41: (24, 24),
    42: (24, 25),
    43: (25, 25),
    44: (25, 26),
    45: (26, 26),
    46: (26, 27),
    47: (27, 27),
    48: (28, 28),
    49: (28, 29),
    50: (29, 29),
    51: (29, 30),
    52: (30, 30),
    53: (31, 31),
    54: (31, 32),
    55: (32, 32),
    56: (32, 33),
    57: (33, 33),
    58: (33, 34),
    59: (34, 34),
    60: (35, 35),
    61: (35, 36),
    62: (36, 36),
    63: (36, 37),
    64: (37, 37),
    65: (38, 38),
    66: (38, 39),
    67: (39, 39),
    68: (39, 40),
    69: (40, 40),
    70: (40, 41),
    71: (41, 41),
    72: (42, 42),
    73: (42, 43),
    74: (43, 43),
    75: (43, 44),
    76: (44, 44),
    77: (45, 45),
    78: (45, 46),
    79: (46, 46),
    80: (46, 47),
    81: (47, 47),
    82: (47, 48),
    83: (48, 48),
    84: (49, 49),
    85: (49, 50),
    86: (50, 50),
    87: (50, 51),
    88: (51, 51),
    89: (52, 52),
    90: (52, 53),
    91: (53, 53),
    92: (53, 54),
    93: (54, 54),
    94: (54, 55),
    95: (55, 55),
    96: (56, 56),
    97: (56, 57),
    98: (57, 57),
    99: (57, 58),
    100: (58, 58),
    101: (59, 59),
    102: (59, 60),
    103: (60, 60),
    104: (60, 61),
    105: (61, 61),
    106: (61, 62),
    107: (62, 62),
    108: (63, 63),
    109: (63, 64),
    110: (64, 64),
    111: (64, 65),
    112: (65, 65),
    113: (66, 66),
    114: (66, 67),
    115: (67, 67),
    116: (67, 68),
    117: (68, 68),
    118: (68, 69),
    119: (69, 69),
    120: (70, 70),
    121: (70, 71),
    122: (71, 71),
    123: (71, 72),
    124: (72, 72),
    125: (73, 73),
    126: (73, 74),
    127: (74, 74)}

def is_white(midi):
    nearest = MIDI_TO_NEAREST_WHITES[midi]
    return nearest[0] == nearest[1]

def key_pattern(midi_set):
    # [0#](:N#M)*
    s = sorted(list(midi_set))
    if len(s) == 0:
        return ""

    result = [ ("#", "W")[is_white(s[0])] ]
    for i in range(1, len(s)):
        result.append(
            str(MIDI_TO_NEAREST_WHITES[s[i]][1] - MIDI_TO_NEAREST_WHITES[s[i-1]][1]) +
            ("#", "W")[is_white(s[i])])
    return ":".join(result)


def pitch_class_set(midi_set):
    return {pitch.Pitch(x).pitchClass for x in midi_set}


def extract_core_events(part, warnings_set):
    """

    Parameters
    ----------
    part
    warnings_set

    Returns
    -------
    event["pitch_set"]
    event["beat"]: fraction
    event["<prefix>duration_symbolic"]
    event["velocity"]
    TODO: articulation ...
    """
    p = recurseNotationAndRests(part.stripTies()).flatten()
    result = []
    for x in p:
        duration = x.duration.quarterLength
        # define current_quarter_ms + last_bpm_anchor_ms
        if duration == 0:
            continue
        if not x.isRest:
            event = {}

            pitch_set = set()
            if 'Note' in x.classes:
                pitch_set.add(x.pitch.midi)
            elif 'Chord' in x.classes:
                for i in range(len(x)):
                    pitch_set.add(x[i].pitch.midi)
            event["finger_set"] = {
                a.fingerNumber for a in x.articulations if a.isClassOrSubclass(
                    (m21.articulations.Fingering,))}
            event["pitch_set"] = pitch_set
            event["pitch_class_set"] = pitch_class_set(pitch_set)
            event["key_pattern"] = key_pattern(pitch_set)
            event["onset_beats"] = Fraction(str(p.elementOffset(x)))
            event["velocity"] = x.volume.cachedRealized
            # TODO: offsets
            result.append(event)
        # else:
        # manipulate previous duration
        # append rests
        #    if len(filtered) > 0:
        #        filtered[len(filtered) - 1][2] += x[2]
    return result


def estimate_absolute_time(part):
    tempo_marks = list(part.recurse().getElementsByClass(m21.tempo.MetronomeMark))
    tempo_marks_by_bars = {
        find_parent(x, m21.stream.Measure).offset: x
        for x in tempo_marks
    }
    # TODO: default value/guess?
    if len(tempo_marks_by_bars) == 0:
        raise Exception('Missed tempo mark')
    #event["onset_ms"] = (p.elementOffset(x) - last_bpm_offset) * current_quarter_ms + last_bpm_ms


def merge_part_events(part_events):
    return sorted(part_events, key=lambda x:x["onset_beats"])


def merge_parts(rh_prefix, rh_events, lh_prefix, lh_events):
    rh_index = 0
    lh_index = 0
    result = []
    while rh_index < len(rh_events) or lh_index < len(lh_events):
        if rh_index == len(rh_events):
            next = lh_events[lh_index]
            prefix = lh_prefix
            lh_index += 1
        elif lh_index == len(lh_events):
            next = rh_events[rh_index]
            prefix = rh_prefix
            rh_index += 1
        elif lh_events[lh_index]["onset_beats"] <= rh_events[rh_index]["onset_beats"]:
            next = lh_events[lh_index]
            prefix = lh_prefix
            lh_index += 1
        else:
            next = rh_events[rh_index]
            prefix = rh_prefix
            rh_index += 1

        joined = {}
        joined["onset_beats"] = next["onset_beats"]
        joined[prefix + "finger_set"] = next["finger_set"]
        joined[prefix + "pitch_set"] = next["pitch_set"]
        joined[prefix + "pitch_class_set"] = next["pitch_class_set"]
        joined[prefix + "key_pattern"] = next["key_pattern"]
        joined["finger_set"] = next["finger_set"].copy()
        joined["pitch_set"] = next["pitch_set"].copy()
        joined["pitch_class_set"] = next["pitch_class_set"].copy()
        joined["key_pattern"] = next["key_pattern"]
        # TODO: add other hand-related keys when they appears.

        if len(result) == 0 or result[-1]["onset_beats"] < next["onset_beats"]:
            result.append(joined)
        else:
            # merge
            last = result[-1]
            last["finger_set"].update(joined["finger_set"])
            last["pitch_set"].update(joined["pitch_set"])
            last["pitch_class_set"].update(joined["pitch_class_set"])
            last["key_pattern"] = key_pattern(last["pitch_set"])
            last[prefix + "finger_set"] = joined[prefix + "finger_set"]
            last[prefix + "pitch_set"] = joined[prefix + "pitch_set"]
            last[prefix + "pitch_class_set"] = joined[prefix + "pitch_class_set"]
            last[prefix + "key_pattern"] = joined[prefix + "key_pattern"]

    return result


def pitch_set_id(midi_set):
    s = sorted(list(midi_set))
    return " ".join([str(x) for x in s])


def get_edges(events, id_attr):
    res = Counter()
    if len(events) >= 2:
        node1 = pitch_set_id(events[0][id_attr])
        for ev in events[1:]:
            node2 = pitch_set_id(ev[id_attr])
            res[node1 + '-' + node2] += 1
            node1 = node2
    return res


def get_key_pattern_edges(events):
    res = Counter()
    if len(events) >= 2:
        node1 = events[0]["key_pattern"]
        lowest1 = min(events[0]["pitch_set"])
        for ev in events[1:]:
            node2 = ev["key_pattern"]
            lowest2 = min(ev["pitch_set"])
            distance = MIDI_TO_NEAREST_WHITES[lowest1][0] - MIDI_TO_NEAREST_WHITES[lowest2][0]
            if distance > 0:
                distance = '+' + str(distance)
            else:
                distance = str(distance)
            res[node1 + ' ' + distance + " " + node2] += 1
            node1 = node2
            lowest1 = lowest2
    return res



def pitch_distributions(events, bpm):
    pitch_counter = Counter()
    pitch_class_counter = Counter()
    pitch_set_counter = Counter()
    key_pattern_counter = Counter()
    pitch_class_set_counter = Counter()
    pitch_intervals = []
    time_intervals = []
    previous_event = None
    pitch_sets = []
    finger_sets = []
    for event in events:
        pitch_list = list(event["pitch_set"])
        event["average_pitch"] = round(int(np.mean(pitch_list)))
        pitch_sets.append(sorted(pitch_list))
        finger_sets.append(sorted(list(event["finger_set"])))
        pitch_counter.update([str(x) for x in event["pitch_set"]])
        pitch_class_counter.update([str(x) for x in event["pitch_class_set"]])
        pitch_set_counter[pitch_set_id(event["pitch_set"])] += 1
        pitch_class_set_counter[pitch_set_id(event["pitch_class_set"])] += 1
        key_pattern_counter[event["key_pattern"]] += 1
        if previous_event is not None:
            pitch_intervals.append(event["average_pitch"] - previous_event["average_pitch"])
            time_intervals.append((event["onset_beats"] - previous_event["onset_beats"]) * 60000.0 / bpm)
        previous_event = event

    pitch_set_edges = get_edges(events, "pitch_set")
    pitch_class_set_edges = get_edges(events, "pitch_class_set")
    key_pattern_edges = get_key_pattern_edges(events)
    return {
        "pitch_sets": pitch_sets,
        "finger_sets": finger_sets,
        "pitch_intervals": pitch_intervals,
        "time_intervals": time_intervals,
        "pitch_frequencies": dict(pitch_counter),
        # temporary name
        "pitch_entropy": entropy(list(pitch_counter.values()), base=2),
        "pitch_class_frequencies": dict(pitch_class_counter),
        "pitch_class_entropy": entropy(list(pitch_class_counter.values()), base=2),
        "pitch_set_frequencies": dict(pitch_set_counter),
        "pitch_set_entropy": entropy(list(pitch_set_counter.values()), base=2),
        "key_pattern_frequencies": dict(key_pattern_counter),
        "key_pattern_entropy": entropy(list(key_pattern_counter.values()), base=2),
        "pitch_class_set_frequencies": dict(pitch_class_set_counter),
        "pitch_class_set_entropy": entropy(list(pitch_class_set_counter.values()), base=2),
        "pitch_set_entropy_rate": entropy(list(pitch_set_edges.values()), base=2),
        "pitch_set_lz": lz_complexity([pitch_set_id(x["pitch_set"]) for x in events]),
        "pitch_class_set_entropy_rate": entropy(list(pitch_class_set_edges.values()), base=2),
        "key_pattern_entropy_rate": entropy(list(key_pattern_edges.values()), base=2),
    }


def performance_chain(rh_part, lh_part, warnings_set):
    """
    Parameters
    ----------
    rh_part
    lh_part
    warnings_set

    Returns
    -------
    event["rh_pitch_set"]
    event["lh_pitch_set"]
    event["beat"]
    event["seconds"]
    event["rh_duration_symbolic"]
    event["rh_duration_seconds"]
    event["lh_duration_symbolic"]
    event["lh_duration_seconds"]
    event["metric_position"]
    event["metric_level"]
    event["velocity"]
    event["rh_syncope_or_regular"]
    event["lh_syncope_or_regular"]
    """

    rh_events = extract_core_events(rh_part, warnings_set)
    lh_events = extract_core_events(lh_part, warnings_set)
    rh_events = merge_part_events(rh_events)
    lh_events = merge_part_events(lh_events)
    return rh_events, lh_events, merge_parts("rh_", rh_events, "lh_", lh_events)
    # TODO: metronome and meter related events
    # TODO: extract partial graphs: pitch, hands, velocity, metric position, ...

if __name__ == '__main__':
    from common_utils import safe_convert
    musicxml_file = "../../Piano/Piano2021/2021.3.Hound_Dog.musicxml.xml"
    score = safe_convert(musicxml_file, forceSource=True)
    # Detect fingering.
    compute_piece_argnn(score)

    lh_part = score.getElementsByClass(stream.Part)[-1]
    rh_part = score.getElementsByClass(stream.Part)[-2]
    warnings_set = set()
    rh_events, lh_events, joint_events = performance_chain(rh_part, lh_part, warnings_set)
    print(lh_events)
    print(pitch_distributions(rh_events, 120))
    print(pitch_distributions(lh_events, 120))
    print(pitch_distributions(joint_events, 120))

