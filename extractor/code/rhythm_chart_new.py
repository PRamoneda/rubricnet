from common_utils import recurseNotation
from rhythm_chart import find_voice_id_and_measure_offset, quarter_ms,\
    SongRhythmInfo, rhythm_stream, extract_metr_events_and_first_measure_number
from fractions import Fraction
import music21 as m21
from music21.tempo import MetronomeMark
from collections import namedtuple
from sortedcontainers import SortedList
from scipy.stats import hmean
import numpy as np
import itertools
import collections
import math

OnsetEvent = namedtuple("OnsetEvent", ['offset', 'measure_offset', 'beat', 'pitch_set', 'duration'])
BarPosWeightEvent = namedtuple("BarPosWeightEvent", ['measure_number', 'beat', 'error_proof_pos', 'weight'])


def voice_onsets_map(part):
    voice_map = {}
    for x in recurseNotation(part.stripTies(matchByPitch=False)):
        voice_id, measure_offset = find_voice_id_and_measure_offset(x)
        pitch_set = set()
        if 'Note' in x.classes:
            pitch_set.add(x.pitch.midi)
        elif 'Chord' in x.classes:
            for i in range(len(x)):
                pitch_set.add(x[i].pitch.midi)
        if voice_id in voice_map:
            l = voice_map.get(voice_id)
        else:
            l = []
            voice_map[voice_id] = l
        l.append(OnsetEvent(Fraction(measure_offset) + x.offset, measure_offset, x.offset, pitch_set,  x.duration.quarterLength))
    return voice_map


def merge_lists(event_lists, warnings_set):
    result = []
    ListNumberAndOffset = namedtuple("ListNumberAndOffset", ["index", "offset"])
    offsets = SortedList(key = lambda x: x.offset)
    positions = [0] * len(event_lists)
    for i in range(len(event_lists)):
        offsets.add(ListNumberAndOffset(i, event_lists[i][positions[i]].offset))

    while len(offsets) > 0:
        i = offsets.pop(0).index
        event = event_lists[i][positions[i]]
        if len(result) == 0 or result[-1].offset < event.offset:
            result.append(event)
        elif result[-1].offset == event.offset:
            result[-1].pitch_set.update(event.pitch_set)
            # TODO: what we should do?
            if event.duration > result[-1].duration:
                result[-1] = OnsetEvent(result[-1].offset, result[-1].measure_offset, result[-1].beat, result[-1].pitch_set, event.duration)
        else:
            warnings_set.add("Unordered event list")
        positions[i] += 1
        if positions[i] < len(event_lists[i]):
            offsets.add(ListNumberAndOffset(i, event_lists[i][positions[i]].offset))
    return result


def syncopation_statistics_for_lists(event_lists, offset_to_effective_number, segments_by_bar, segments_by_meter_tempo, warnings_set, decimation=True):
    ioi_durations = []
    event_lists = [x for x in event_lists if len(x) > 0]
    # TODO: main counting loop. Syncopes, Playing speed, Stamina (average pitch onsets per bar)
    l = merge_lists(event_lists, warnings_set)
    bar_pos_weight = []
    time_spans_by_bar = {}
    stamina_by_bar = {}

    for i in range(len(l)):
        onsetEvent = l[i]
        measure_number = offset_to_effective_number[onsetEvent.measure_offset]
        if i == (len(l) - 1):
            duration = onsetEvent.duration
            next_measure_number = measure_number + 1
        else:
            if type(l[i+1].offset) is Fraction or type(onsetEvent.offset) is Fraction:
                duration = Fraction(l[i + 1].offset) - Fraction(onsetEvent.offset)
            else:
                duration = l[i + 1].offset - onsetEvent.offset
            next_measure_number = offset_to_effective_number[l[i+1].measure_offset]
        ioi_durations.append(duration)

        # Update stamina. TODO: move to separated function
        stamina = stamina_by_bar.get(measure_number, 0)
        stamina += len(onsetEvent.pitch_set)
        stamina_by_bar[measure_number] = stamina

        # segment and syncopes
        segment = segments_by_bar[measure_number]
        error_proof_pos, weight = segment.metrical_hierarchy.get_position_syncope_weight(
            onsetEvent.beat, duration, warnings_set, measure_number)
        bar_pos_weight.append(BarPosWeightEvent(measure_number, onsetEvent.beat, error_proof_pos, weight))

        # Update time spans. TODO: move to separated function
        if next_measure_number - measure_number <= 1:
            span = segment.metrical_hierarchy.quarter_duration_ms * duration
        else:
            # approximate phrase end
            span = segment.metrical_hierarchy.quarter_duration_ms * onsetEvent.duration
        if measure_number not in time_spans_by_bar:
            time_spans_by_bar[measure_number] = []
        time_spans_by_bar[measure_number].append(float(span))

    # Syncope counting
    syncopes = []
    for x in bar_pos_weight:
        if x.weight is None:
            continue
        segment = segments_by_bar[x.measure_number]
        if x.weight > 0:
            segment.syncopes_histogram[x.error_proof_pos] += 1
            syncopes.append({"bar":x.measure_number, "beat":Fraction(x.beat), "pos":x.error_proof_pos, "weight":x.weight})
        else:
            segment.regulars_histogram[x.error_proof_pos] += 1
    if decimation:
        for segment in segments_by_meter_tempo.values():
            segment.decimate_empty_layers()

    # TODO: try other averaging
    stamina = hmean(list(stamina_by_bar.values()))
    stamina_new = sum(list(stamina_by_bar.values()))

    all_spans = np.fromiter(itertools.chain.from_iterable(time_spans_by_bar.values()), float)
    all_spans2 = all_spans * all_spans
    avg_playing_speed_per_measure = {k:1000.0/np.mean(v) for k, v in time_spans_by_bar.items()}
    avg_playing_speed_per_score = 1000.0 / np.mean(all_spans)
    avg_playing_speed2_per_measure = {k:1000.0/np.mean(np.power(v, 2)) for k, v in time_spans_by_bar.items()}
    avg_playing_speed2_per_score = 1000.0 / np.mean(all_spans2)
    return SongRhythmInfo(
        list(segments_by_meter_tempo.values()),
        syncopes,
        avg_playing_speed_per_measure, avg_playing_speed_per_score,
        avg_playing_speed2_per_measure, avg_playing_speed2_per_score,
        stamina, stamina_new, ioi_quarter_values=ioi_durations), l, all_spans


def syncopation_statistics_for_part(part, offset_to_effective_number, segments_by_bar, segments_by_meter_tempo, warnings_set, decimation=True):
    voice_map = voice_onsets_map(part)
    return syncopation_statistics_for_lists(list(voice_map.values()), offset_to_effective_number, segments_by_bar, segments_by_meter_tempo, warnings_set, decimation)


if __name__ == '__main__':
    from common_utils import safe_convert
    from score_analyser.raw_data_extractors import KeysFeatureExtractor
    from copy import deepcopy
    from score_analyser.raw_data_extractors import propagate_metr_events_to_parts

    warnings_set = set()
    musicxml_file = "../../../CIPI_symbolic/xmander_files/5171003.musicxml"
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
    print(KeysFeatureExtractor().extract_raw_data_map_from_streams(
        m21_stream,
        m21_stream_with_metronome_marks,
        m21_stream_with_metronome_marks,
        warnings_set))
