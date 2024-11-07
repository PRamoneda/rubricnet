import music21 as m21

def clefs_statistics(part, offset_to_effective_number):
    clefs_by_bar = {}
    for measure in part.recurse(classFilter=m21.stream.Measure):
        effective_bar_number = offset_to_effective_number[measure.offset]
        clefs = list(measure.recurse(classFilter='Clef'))
        if len(clefs) > 0:
            clefs_by_bar[effective_bar_number] = clefs[0].__class__.__name__
    return clefs_by_bar

def key_signatures_statistics(part, offset_to_effective_number):
    key_signature_by_bar = {}
    for measure in part.recurse(classFilter=m21.stream.Measure):
        effective_bar_number = offset_to_effective_number[measure.offset]
        signatures = list(measure.recurse(classFilter='KeySignature'))
        if len(signatures) > 0:
            key_signature_by_bar[effective_bar_number] = signatures[0].sharps
    return key_signature_by_bar