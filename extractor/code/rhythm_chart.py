import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import music21 as m21
from music21.tempo import MetronomeMark
from sortedcontainers import SortedDict
import math

from common_utils import find_parent, recurseNotationAndRests, extract_metr_events_and_first_measure_number, \
    build_offset_to_effective_bar_number_map
from fractions import Fraction

SINGLETON_VOICE = 'singleton voice'

def find_voice_id_and_measure_offset(el):
    """
    Returns the first parent of el of type class_type. Both el and class_type must be valid music21 objects
    If class_type does not exist in the parent chain of el, the outermost object will be returned
    """
    temp = el
    measure_offset = None
    voice_id = None
    while temp is not None:
        if measure_offset is None and isinstance(temp, m21.stream.Measure):
            measure_offset = temp.getOffsetBySite(temp.activeSite)
        if voice_id is None and isinstance(temp, m21.stream.Voice):
            voice_id = temp.id
        if voice_id is not None and measure_offset is not None:
            break
        temp = temp.activeSite

    if measure_offset is None:
        raise Exception("Can't identify measure number")
    if voice_id is None:
        voice_id = SINGLETON_VOICE
    return voice_id, measure_offset


def dissolve_rests(event_list):
    filtered = []
    # append rests
    for x in event_list:
        if x[2] == 0:
            continue
        if not x[3]:
            filtered.append(x[:-1])
        else:
            if len(filtered) > 0:
                filtered[len(filtered) - 1][2] += Fraction(x[2])
    return filtered


def _offset(x):
    return x[0] + x[1]

def merge_voices(voices):
    result = []
    offsets = SortedDict()
    positions = [0] * len(voices)
    for i in range(len(voices)):
        offsets[_offset(voices[i][positions[i]])] = i

    while len(offsets) > 0:
        i = offsets.popitem(0)[1]
        result.append(voices[i][positions[i]])
        positions[i] += 1
        if positions[i] < len(voices[i]):
            offsets[_offset(voices[i][positions[i]])] = i
    return result


def rhythm_stream(part):
    """
    Parameters
    ----------
    part - music21 part

    Returns list of triplets: (measure_offset, beatPosition, quarterLength)
    -------

    """
    p = recurseNotationAndRests(part.stripTies(matchByPitch=False))
    l = [[*find_voice_id_and_measure_offset(x), x.offset, x.duration.quarterLength, x.isRest] for x in p]
    voice_map = {}
    for item in l:
        if item[0] not in voice_map:
            voice_map[item[0]] = []
        voice_map[item[0]].append(item[1:])
    # dissolve rests before merge
    voice_map_filtered = {k:dissolve_rests(v) for k, v in voice_map.items()}
    voice_map_filtered = {k:v for k, v in voice_map_filtered.items() if len(v) > 0}
    return merge_voices(list(voice_map_filtered.values()))


EPS = 0.1


class MetricalHierarchy:
    def __init__(self, ratio_string, beats_per_bar, bar_unit_label, quarter_duration_ms):
        self.ratio_string = ratio_string
        self.quarter_duration_ms = quarter_duration_ms
        self.position_levels = [0]
        self.level_unit_beats = [beats_per_bar]
        self.level_unit_labels = [bar_unit_label]
        self.atoms_per_quarter = 0

    def get_quarter_bpm(self):
        return round(60000 / self.quarter_duration_ms)

    def add_level(self, multiple):
        beats_per_level = self.level_unit_beats[-1] / multiple
        label = RHYTHM_VALUE_CHARACTERS[beats_per_level]
        self.level_unit_beats.append(beats_per_level)
        self.level_unit_labels.append(label)
        new_position_levels = []
        level = len(self.level_unit_beats) - 1
        for x in self.position_levels:
            new_position_levels.append(x)
            for i in range(1, multiple):
                new_position_levels.append(level)
        self.position_levels = new_position_levels
        self.atoms_per_quarter = round(1 / beats_per_level)
        return self

    def get_min_syncopated_duration(self, pos, level):
        min_syncopated_duration = self.level_unit_beats[level]
        for next_pos in range(pos + 1, len(self.position_levels)):
            next_level = self.position_levels[next_pos]
            if next_level > level:
                continue
            elif next_level == level:
                min_syncopated_duration += self.level_unit_beats[next_level]
            else:
                break
        return min_syncopated_duration

    def position_labels(self):
        return np.array([self.level_unit_labels[x] for x in self.position_levels])

    def position_syncopation_level(self):
        res = np.empty(len(self.position_levels))
        for i in range(len(self.position_levels)):
            delta = 0
            if self.position_levels[i] > 0:
                for j in range(i, len(self.position_levels)):
                    if (self.position_levels[i] - self.position_levels[j]) > 0:
                        delta = self.position_levels[i] - self.position_levels[j]
                        break
                if delta == 0:
                    delta = self.position_levels[i]
            res[i] = delta
        return res

    def level_unit_durations_ms(self):
        return self.quarter_duration_ms * np.array(self.level_unit_beats)

    def get_position_syncope_weight(self, beat, duration, warnings_set, measure):
        """
        Parameters
        ----------
        beat position in the bar
        duration duration

        Returns (position, syncope weight)
        -------

        """
        beat *= self.atoms_per_quarter
        pos = round(beat)
        if abs(beat - pos) > EPS:
            # tolerate numerical error, but ignore too fine metric levels
            return None, None
        error_proof_pos = pos % len(self.position_levels)
        if error_proof_pos != pos:
            print('Measure: ', measure, 'Beat: ', beat, "Duration: ", duration)
            warnings_set.add("Measure longer than defined by time signature is found")

        level = self.position_levels[error_proof_pos]
        min_syncopated_duration = self.get_min_syncopated_duration(error_proof_pos, level)
        # duration is strictly greater (assuminig numeric error)
        if error_proof_pos > 0 and duration - min_syncopated_duration > EPS:
            # syncope!
            displaced_pos = int((error_proof_pos + round(min_syncopated_duration * self.atoms_per_quarter))) % len(self.position_levels)
            weight = self.position_levels[error_proof_pos] - self.position_levels[displaced_pos]
            return error_proof_pos, weight
        else:
            # nope
            return error_proof_pos, 0

RHYTHM_VALUE_CHARACTERS = {
    # whole (4 quarters)
    Fraction(4):'\U0001D15D',
    # dotted half (3 quarters)
    Fraction(3):'\U0001D15E \U0001D16D',
    # half (2 quarters)
    Fraction(2): '\U0001D15E',
    # dotted quarter
    Fraction(3, 2): '\U0001D15F \U0001D16D',
    # quarter
    Fraction(1): '\U0001D15F',
    # dotted eight
    Fraction(3, 4): '\U0001D160 \U0001D16D',
    # eight (half quarter)
    Fraction(1, 2): '\U0001D160',
    # sixteenth (quarter of quarter)
    Fraction(1, 4): '\U0001D161',
    # 32nd (eight of quarter)
    Fraction(1, 8): '\U0001D162',
    # 64th (sixteenth of quarter)
    Fraction(1, 16): '\U0001D163'
}


class Meter4by4(MetricalHierarchy):
    # levels
    # 0 - whole note
    # 1 - half
    # 2 - quarter
    # 3 - eight
    # 4 - sixteenth
    # 5 - 32nd

    def __init__(self, quarter_duration_ms):
        super().__init__("4/4", 4, '\U0001D15D', quarter_duration_ms)
        self.add_level(2).add_level(2).add_level(2).add_level(2).add_level(2)


class Meter2by4(MetricalHierarchy):
    # levels
    # 1 - half
    # 2 - quarter
    # 3 - eight
    # 4 - sixteenth
    # 5 - 32nd

    def __init__(self, quarter_duration_ms):
        super().__init__("2/4", 2, '\U0001D15E', quarter_duration_ms)
        self.add_level(2).add_level(2).add_level(2).add_level(2)


class AnythingOverBinaryPlainMeter(MetricalHierarchy):
    # levels
    # 0: M x 1/N
    # 1: 1/N
    # 2: 1/(2*N)
    # ...
    # (5 - Log_2(N)) + 1: 32nd
    # for N<=8: 64nd

    def __init__(self, M, N, quarter_duration_ms):
        #def __init__(self, ratio_string, beats_per_bar, bar_unit_label, quarter_duration_ms):
        super().__init__(str(M) + "/" + str(N),
                         Fraction(4 * M, N),
                         str(M) + "x" + RHYTHM_VALUE_CHARACTERS[Fraction(4, N)],
                         quarter_duration_ms)
        self.add_level(M)
        if N>=4:
            # up to 32nd
            for i in range(5 - (math.frexp(N)[1] - 1)):
                self.add_level(2)
        else:
            # up to 64th
            for i in range(6 - (math.frexp(N)[1] - 1)):
                self.add_level(2)


class Meter2by2(MetricalHierarchy):
    # levels
    # 0 - whole note
    # 1 - half
    # 2 - quarter
    # 3 - eight
    # 4 - sixteenth

    def __init__(self, quarter_duration_ms):
        super().__init__("2/2", 4, '\U0001D15D', quarter_duration_ms)
        self.add_level(2).add_level(2).add_level(2).add_level(2)


class Meter3by4(MetricalHierarchy):
    # levels
    # 0 - dotted half
    # 1 - quarter
    # 2 - eight
    # 3 - sixteenth
    # 4 - 32nd

    def __init__(self, quarter_duration_ms):
        super().__init__("3/4", 3, '\U0001D15E \U0001D16D', quarter_duration_ms)
        self.add_level(3).add_level(2).add_level(2).add_level(2)


class Meter3by2(MetricalHierarchy):
    # levels
    # 0 - dotted whole
    # 1 - half
    # 2 - quarter
    # 3 - eight
    # 4 - sixteenth
    # 5 - 32nd

    def __init__(self, quarter_duration_ms):
        super().__init__("3/2", 6, '\U0001D15D \U0001D16D', quarter_duration_ms)
        self.add_level(3).add_level(2).add_level(2).add_level(2).add_level(2)


class Meter6by8(MetricalHierarchy):
    # levels
    # 0 - dotted half
    # 1 - dotted quarter
    # 3 - eight
    # 3 - sixteenth
    # 4 - 32nd

    def __init__(self, quarter_duration_ms):
        super().__init__("6/8", 3, '\U0001D15E \U0001D16D', quarter_duration_ms)
        self.add_level(2).add_level(3).add_level(2).add_level(2)


class Meter6by16(MetricalHierarchy):
    # levels
    # 0 - dotted quarter
    # 1 - eight
    # 2 - sixteenth
    # 3 - 32nd
    # 4 - 64th

    def __init__(self, quarter_duration_ms):
        super().__init__("6/8", 1.5, '\U0001D15F \U0001D16D', quarter_duration_ms)
        self.add_level(2).add_level(3).add_level(2).add_level(2)


class Meter6by4(MetricalHierarchy):
    # kind of a cut time for 6/8.
    # levels
    # 0 - dotted whole
    # 2 - dotted half
    # 3 - quarter
    # 4 - eighth
    # 5 - sixteenth

    def __init__(self, quarter_duration_ms):
        # There's strange bug in Bravura/Matplotlib: after dotted whole note, some
        # lables are corrupted, unless we insert something tall in the beginning.
        super().__init__("6/4", 6, '\U0001D100 \U0001D15D \U0001D16D', quarter_duration_ms)
        self.add_level(2).add_level(3).add_level(2).add_level(2)


class Meter12by8(MetricalHierarchy):
    # levels
    # 0 - dotted whole
    # 2 - dotted half
    # 3 - dotted quarter
    # 4 - eighth
    # 5 - sixteenth
    # 6 - 32nd

    def __init__(self, quarter_duration_ms):
        # There's strange bug in Bravura/Matplotlib: after dotted whole note, some
        # lables are corrupted, unless we insert something tall in the beginning.
        super().__init__("12/8", 6, '\U0001D100 \U0001D15D \U0001D16D', quarter_duration_ms)
        self.add_level(2).add_level(2).add_level(3).add_level(2).add_level(2)


class Meter12by16(MetricalHierarchy):
    # levels
    # 2 - dotted half
    # 3 - dotted quarter
    # 4 - dotted quarter
    # 5 - sixteenth
    # 6 - 32nd

    def __init__(self, quarter_duration_ms):
        # There's strange bug in Bravura/Matplotlib: after dotted whole note, some
        # lables are corrupted, unless we insert something tall in the beginning.
        super().__init__("12/16", 3, '\U0001D15E \U0001D16D', quarter_duration_ms)
        self.add_level(2).add_level(2).add_level(3).add_level(2).add_level(2)


class Meter9by16(MetricalHierarchy):
    # levels
    # 1 - half+16th
    # 2 - dotted eight
    # 3 - sixteenth
    # 4 - 32nd
    # 5 - 64th

    def __init__(self, quarter_duration_ms):
        # There's strange bug in Bravura/Matplotlib: after dotted whole note, some
        # lables are corrupted, unless we insert something tall in the beginning.
        super().__init__("9/16", 2.25, '\U0001D15E \U0001D16D', quarter_duration_ms)
        self.add_level(3).add_level(3).add_level(2).add_level(2)


class Meter5by4(MetricalHierarchy):
    # Temporary decision for 5/4.
    # No level distinction betwee 2-5 quarters
    # levels
    # 0 - whole + quarter
    # 1 - quarter
    # 2 - eighth
    # 3 - sixteenth
    # 4 - 32nd

    def __init__(self, quarter_duration_ms):
        # There's strange bug in Bravura/Matplotlib: after dotted whole note, some
        # lables are corrupted, unless we insert something tall in the beginning.
        super().__init__("5/4", 5, '\U0001D15D \U0001D15F', quarter_duration_ms)
        self.add_level(5).add_level(2).add_level(2).add_level(2)


class Meter9by8(MetricalHierarchy):
    # Temporary decision for 9/8.
    # 0 - whole + eighth
    # 3 - dotted quarter
    # 4 - eighth
    # 5 - sixteenth
    # 6 - 32nd

    def __init__(self, quarter_duration_ms):
        # There's strange bug in Bravura/Matplotlib: after dotted whole note, some
        # lables are corrupted, unless we insert something tall in the beginning.
        super().__init__("9/8", 4.5, '\U0001D15D \U0001D160', quarter_duration_ms)
        self.add_level(3).add_level(3).add_level(2).add_level(2)


class SegmentInfo:
    def __init__(self, metrical_hierarchy):
        self.metrical_hierarchy = metrical_hierarchy
        self.syncopes_histogram = np.zeros(len(metrical_hierarchy.position_levels))
        self.regulars_histogram = np.zeros(len(metrical_hierarchy.position_levels))
        self.syncopation_strengths = metrical_hierarchy.position_syncopation_level()
        self.position_levels = np.array(metrical_hierarchy.position_levels)
        self.labels = metrical_hierarchy.position_labels()
        self.bars = []

    def decimate_empty_layers(self):
        empty_layers = set(range(
            len(self.metrical_hierarchy.level_unit_beats)))
        for i in range(len(self.syncopes_histogram)):
            if (self.syncopes_histogram[i] > 0 or self.regulars_histogram[i] > 0) and self.position_levels[i] in empty_layers:
                empty_layers.remove(self.position_levels[i])
        salient_layers = set(range(len(self.metrical_hierarchy.level_unit_beats)))
        for i in range(len(self.metrical_hierarchy.level_unit_beats) - 1, -1, -1):
            if i in empty_layers:
                salient_layers.remove(i)
            else:
                break
        non_empty_layers = [self.position_levels[x] in salient_layers for x in range(len(self.syncopes_histogram))]

        # TODO: decimation by salience (?)
        self.syncopes_histogram = self.syncopes_histogram[non_empty_layers]
        self.regulars_histogram = self.regulars_histogram[non_empty_layers]
        self.syncopation_strengths = self.syncopation_strengths[non_empty_layers]
        self.position_levels = self.position_levels[non_empty_layers]
        self.labels = self.labels[non_empty_layers]

    def to_serializable_dict(self):
        result = {}
        result['metrical_hierarchy'] = self.metrical_hierarchy.ratio_string
        result['quarter_duration_ms'] = self.metrical_hierarchy.quarter_duration_ms
        result['syncopes_histogram'] = self.syncopes_histogram.tolist()
        result['regulars_histogram'] = self.regulars_histogram.tolist()
        result['syncopation_strengths'] = self.syncopation_strengths.tolist()
        result['position_levels'] = self.position_levels.tolist()
        result['labels'] = self.labels.tolist()
        result['bars'] = self.bars
        return result


def segment_info_from_dict(a_dict, warnings_set):
    metrical_hierarchy = create_metrical_hierarchy(
        a_dict['metrical_hierarchy'],
        a_dict['quarter_duration_ms'],
        warnings_set)
    result = SegmentInfo(metrical_hierarchy)
    result.metrical_hierarchy = metrical_hierarchy
    result.syncopes_histogram = np.array(a_dict['syncopes_histogram'])
    result.regulars_histogram = np.array(a_dict['regulars_histogram'])
    result.syncopation_strengths = np.array(a_dict['syncopation_strengths'])
    result.position_levels = np.array(a_dict['position_levels'])
    result.labels = np.array(a_dict['labels'])
    result.bars = a_dict['bars']
    return result


class SongRhythmInfo:
    def __init__(
            self,
            segments,
            syncopes,
            avg_playing_speed_per_measure, avg_playing_speed_per_score,
            avg_playing_speed2_per_measure, avg_playing_speed2_per_score,
            stamina,
            stamina_new,
            ioi_quarter_values):
        self.segments = segments
        self.syncopes = syncopes
        self.avg_playing_speed_per_measure = avg_playing_speed_per_measure
        self.avg_playing_speed_per_score = avg_playing_speed_per_score
        self.avg_playing_speed2_per_measure = avg_playing_speed2_per_measure
        self.avg_playing_speed2_per_score = avg_playing_speed2_per_score
        self.stamina = stamina
        self.stamina_new = stamina_new
        self.ioi_quarter_values = ioi_quarter_values

    def to_serializable_dict(self):
        result = {}
        result['syncopes'] = [{"bar":x["bar"], "beat":str(x["beat"]), "pos":x["pos"], "weight":x["weight"]} for x in self.syncopes]
        result['segments'] = [x.to_serializable_dict() for x in self.segments]
        result['avg_playing_speed_per_measure'] = self.avg_playing_speed_per_measure
        result['avg_playing_speed_per_score'] = self.avg_playing_speed_per_score
        result['avg_playing_speed2_per_measure'] = self.avg_playing_speed2_per_measure
        result['avg_playing_speed2_per_score'] = self.avg_playing_speed2_per_score
        result['stamina'] = self.stamina
        result['stamina_new'] = self.stamina_new
        result['ioi_quarter_values'] = [str(x) for x in self.ioi_quarter_values]
        return result


def song_rhythm_info_from_dict(a_dict, warnings_set):
    segments = [segment_info_from_dict(x, warnings_set) for x in a_dict['segments']]
    syncopes = [{"bar":x["bar"], "beat":Fraction(x["beat"]), "pos":x["pos"], "weight":x["weight"]} for x in a_dict['syncopes']]
    return SongRhythmInfo(
        segments, syncopes,
        a_dict["avg_playing_speed_per_measure"], a_dict["avg_playing_speed_per_score"],
        a_dict["avg_playing_speed2_per_measure"], a_dict["avg_playing_speed2_per_score"],
        a_dict["stamina"],
        a_dict["stamina_new"],
        [Fraction(x) for x in a_dict["ioi_quarter_values"]])

def quarter_ms(tempo_mark):
    return int(tempo_mark.durationToSeconds(m21.duration.Duration('quarter')) * 1000)

RHYTHM_HIERARCHIES_BY_RATIO_STRING = {
    '4/4':Meter4by4, '2/2':Meter2by2, '3/4':Meter3by4, '6/4':Meter6by4, '6/8':Meter6by8, '12/8':Meter12by8,
    '12/16':Meter12by16, '9/16':Meter9by16, '6/16':Meter6by16,
    '5/4':Meter5by4, "2/4":Meter2by4, "9/8":Meter9by8,
    "3/2":Meter3by2}


def create_metrical_hierarchy(time_signature_string_ratio, quarter_ms, warnings_set):
    if time_signature_string_ratio in RHYTHM_HIERARCHIES_BY_RATIO_STRING:
        return RHYTHM_HIERARCHIES_BY_RATIO_STRING[time_signature_string_ratio](quarter_ms)
    else:
        warnings_set.add("Weird metric hierarchy is created on the fly: " + time_signature_string_ratio)
        # TODO: For even cases, detect actual grouping
        return AnythingOverBinaryPlainMeter(*([int(x) for x in time_signature_string_ratio.split("/")]), quarter_ms)


def create_segment_by_meter_tempo(time_signature_string_ratio, quarter_ms, warnings_set):
    return SegmentInfo(create_metrical_hierarchy(time_signature_string_ratio, quarter_ms, warnings_set))


def rhythm_segments(part, offset_to_effective_number, warnings_set, ignore_tempo_marks=False):
    tempo_marks_by_bars = {}
    signatures_by_bars = {}

    # Iterate through measures and their content.
    # Otherwise (for still unknown reason) active site could be a garbage.
    # TODO: sort it out
    for measure in part.recurse(classFilter=m21.stream.Measure):
        effective_bar_number = offset_to_effective_number[measure.offset]
        tempo_marks = list(measure.recurse(classFilter=m21.tempo.MetronomeMark))
        if len(tempo_marks) > 0:
            tempo_marks_by_bars[effective_bar_number] = tempo_marks[0]
        signatures = list(measure.recurse(classFilter=m21.meter.TimeSignature))
        if len(signatures) > 0:
            signatures_by_bars[effective_bar_number] = signatures[0]

    if len(tempo_marks_by_bars) == 0 or len(signatures_by_bars) == 0:
        raise Exception('Missed tempo mark or time signature')

    first_tempo_n = min(tempo_marks_by_bars.keys())
    first_signature_n = min(signatures_by_bars.keys())

    if ignore_tempo_marks:
        tempo_marks_by_bars = {first_tempo_n: tempo_marks_by_bars[first_tempo_n]}

    meter_tempo = (signatures_by_bars[first_signature_n].ratioString, quarter_ms(tempo_marks_by_bars[first_tempo_n]))
    max_bar = max(offset_to_effective_number.values())
    min_bar = min(offset_to_effective_number.values())
    segments_by_meter_tempo = {}
    segments_by_bar = {}
    for n in range(min_bar, max_bar + 1):
        if n in tempo_marks_by_bars:
            meter_tempo = (meter_tempo[0], quarter_ms(tempo_marks_by_bars[n]))
        if n in signatures_by_bars:
            meter_tempo = (signatures_by_bars[n].ratioString, meter_tempo[1])
        if meter_tempo not in segments_by_meter_tempo:
            segments_by_meter_tempo[meter_tempo] = create_segment_by_meter_tempo(*meter_tempo, warnings_set)
        segments_by_bar[n] = segments_by_meter_tempo[meter_tempo]
        segments_by_bar[n].bars.append(n)

    return segments_by_bar, segments_by_meter_tempo


def syncopation_statistics(part, offset_to_effective_number, segments_by_bar, segments_by_meter_tempo, warnings_set, decimation=True):
    l = rhythm_stream(part)
    # remap to effective bar numbers
    # music21 measure.number is not unique in case of several endings (volta brackets),
    # and in case of repeat expansion.
    #
    l = [[offset_to_effective_number[x[0]], x[1], x[2]] for x in l]

    bar_pos_weight = [(x[0], x[1], *segments_by_bar[x[0]].metrical_hierarchy.get_position_syncope_weight(x[1], x[2], warnings_set, x[0])) for x in l]

    # measures = part.getElementsByClass(Measure)
    syncopes = []

    for x in bar_pos_weight:
        if x[3] is None:
            continue
        segment = segments_by_bar[x[0]]
        if x[3] > 0:
            segment.syncopes_histogram[x[2]] += 1
            syncopes.append({"bar":x[0], "beat":Fraction(x[1]), "pos":x[2], "weight":x[3]})
        else:
            segment.regulars_histogram[x[2]] += 1
    if decimation:
        for segment in segments_by_meter_tempo.values():
            segment.decimate_empty_layers()

    return SongRhythmInfo(list(segments_by_meter_tempo.values()), syncopes,
                          {}, 0.0,
                          {}, 0.0,
                          0.0, 0.0, [])


def plot_chart(ax_syncopes, ax_regular, segment, fontsize=24):
    my_cmap = plt.get_cmap("Reds")
    x = range(len(segment.regulars_histogram))
    y_max = max(max(segment.syncopes_histogram), max(segment.regulars_histogram)) + 0.5
    ax_syncopes.set_title(
        "%s bpm: %i (%i bars)" % (segment.metrical_hierarchy.ratio_string , segment.metrical_hierarchy.get_quarter_bpm(), len(segment.bars)),
        fontsize=fontsize)
    ax_syncopes.set_ylabel("Syncopes", fontsize=fontsize, x=-0.1, y=0.6)
    ax_syncopes.bar(x, segment.syncopes_histogram, color=my_cmap(segment.syncopation_strengths / max(segment.syncopation_strengths)))
    ax_syncopes.set_ylim([0, y_max])

    my_cmap = plt.get_cmap("cividis")
    fp = font_manager.FontProperties(fname="Bravura.otf")
    ax_regular.bar(x, segment.regulars_histogram,
                color=my_cmap(np.array(segment.position_levels) / max(segment.position_levels)))
    ax_regular.set_ylabel("Regulars", fontsize=fontsize, x=-0.1, y=0.6)
    ax_regular.set_ylim([0, y_max])
    ax_regular.invert_yaxis()

    ax_regular.xaxis.tick_top()
    ax_regular.tick_params(pad=5)
    ax_regular.set_xticks(np.arange(len(segment.labels)))
    # WTF?
    #labels[0] = labels[0]
    #labels[-1] = labels[-1]
    ax_regular.set_xticklabels(
        segment.labels,
        fontproperties=fp,
        ha="center", va="bottom",
        size=fontsize)


def rhythm_charts_by_song_info(song_info, single_plot_width=18, single_plot_height=12, fontsize=24):
    n_segments = len(song_info.segments)
    fig, axes = plt.subplots(
        nrows=2, ncols=n_segments, sharex='col',
        figsize=(single_plot_width * n_segments, single_plot_height))
    if n_segments == 1:
        plot_chart(axes[0], axes[1], song_info.segments[0], fontsize=fontsize)
    else:
        for i in range(n_segments):
            plot_chart(axes[0][i], axes[1][i], song_info.segments[i], fontsize=fontsize)
    return fig


def rhythm_charts(part):
    offset_map = build_offset_to_effective_bar_number_map(part)
    segments_by_bar, segments_by_meter_tempo = rhythm_segments(part, offset_map, set())
    rhythm_charts_by_song_info(syncopation_statistics(
        part,
        offset_map,
        segments_by_bar,
        segments_by_meter_tempo,
        set()))


if __name__ == '__main__':
    from common_utils import safe_convert
    from score_analyser.raw_data_extractors import KeysFeatureExtractor
    from copy import deepcopy
    from score_analyser.raw_data_extractors import propagate_metr_events_to_parts

    warnings_set = set()
    musicxml_file = "../../RP/Keys/Grade3/Edited/KEYS Gr 3 Back In The USSR.xml"
    m21_stream = safe_convert(musicxml_file, forceSource=True)
    m21_stream_with_metronome_marks = deepcopy(m21_stream)
    metronome_marks, time_signatures, m_number = extract_metr_events_and_first_measure_number(
        m21_stream_with_metronome_marks)
    if len(metronome_marks) == 0:
        metronome_marks = {m_number: MetronomeMark(144)}
    m21_stream_with_metronome_marks = propagate_metr_events_to_parts(
        m21_stream_with_metronome_marks, metronome_marks)

    #try:
    #    expanded_stream = m21_stream_with_metronome_marks.expandRepeats()
    #except:
    #    # 2
    #    warnings_set.add("Can't expand score")
    #    expanded_stream = m21_stream_with_metronome_marks
    KeysFeatureExtractor().extract_raw_data_map_from_streams(
        m21_stream,
        m21_stream_with_metronome_marks,
        m21_stream_with_metronome_marks,
        warnings_set)

    #score = propagate_metronome_mark_to_parts(score, metronome_marks)
    #part = score.parts.stream()[0]
    #song_info = syncopation_statistics(part)
    #s = json.dumps(song_info.to_serializable_dict())
    #print(s)
    #rhythm_charts(part)
    #plt.show()
    #song_info2 = song_rhythm_info_from_dict(json.loads(s))
    #rhythm_charts_by_song_info(song_info2)
    #plt.show()
