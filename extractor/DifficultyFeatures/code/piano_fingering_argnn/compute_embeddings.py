import argparse
import os
import sys
import traceback

import music21
import numpy as np
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import piano_fingering_argnn.utils as utils
import piano_fingering_argnn.common as common
import piano_fingering_argnn.seq2seq_model as seq2seq_model
import piano_fingering_argnn.GGCN as GGCN
from common_utils import get_song_map
def choice_model(hand, architecture):
    # load model torch implementation

    model = None
    if architecture == 'ArGNNThumb-s':
        model = seq2seq_model.seq2seq(
            embedding=common.emb_pitch(),
            encoder=seq2seq_model.gnn_encoder(input_size=64),
            decoder=seq2seq_model.AR_decoder(64)
        )
    elif architecture == 'ArLSTMThumb-f':
        model = seq2seq_model.seq2seq(
            embedding=common.only_pitch(),
            encoder=seq2seq_model.lstm_encoder(input=1, dropout=0),
            decoder=seq2seq_model.AR_decoder(64)
        )
    if model is not None:
        assert model is not None, "bad model chosen"
    # load model saved from checkpoint
    model_path = f"{hand}_{architecture}.pth"
    checkpoint = torch.load(os.path.join(os.path.dirname(__file__), f'models/{model_path}'), map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def next_onset(onset, onsets):
    # -1 is a impossible value then there is no next
    ans = '-1'
    hand_onsets = list(set(onsets))
    hand_onsets.sort(key=lambda a: float(a))
    for idx in range(len(hand_onsets)):
        if float(hand_onsets[idx]) > float(onset):
            ans = hand_onsets[idx]
            break
    return ans


def compute_edge_list(onsets, pitchs):
    assert len(onsets) == len(pitchs), "check lenghts"
    edges = []
    for idx, (current_onset, current_pitch) in enumerate(zip(onsets, pitchs)):
        # pdb.set_trace()
        if current_pitch != 0:
            # next labels of right hand
            next_right_hand = next_onset(current_onset, onsets)
            next_labels = [(idx, jdx, "next") for jdx, onset in enumerate(onsets) if
                           onset == next_right_hand and idx != jdx]
            edges.extend(next_labels)
            # onset labels
            onset_edges = [(idx, jdx, "onset") for jdx, onset in enumerate(onsets) if
                           current_onset == onset and idx != jdx]
            edges.extend(onset_edges)
    return edges


def first_note_symmetric(note, from_hand='lh'):
    right2left_pitch_class_symmetric = {
        0: 4,
        1: 2,
        2: 0,
        3: -2,
        4: -4,
        5: -6,
        6: -8,
        7: -10,
        8: -12,
        9: -14,
        10: -16,
        11: -18
    }
    left2right_pitch_class_symmetric = {
        0: 16,
        1: 14,
        2: 12,
        3: 10,
        4: 8,
        5: 6,
        6: 4,
        7: 2,
        8: 0,
        9: -2,
        10: -4,
        11: -6
    }
    pitch_class = note % 12  # 4
    d_oct = (note - 60) // 12  # -1

    if from_hand == 'lh':
        ans = note + left2right_pitch_class_symmetric[pitch_class] - (2 * d_oct * 12) - 24
    else:
        ans = note + right2left_pitch_class_symmetric[pitch_class] - (2 * d_oct * 12)
    return ans

def _surpass_bounds(notes):
    surpass = False
    for n in notes:
        if not (n == 0 or (21 <= n < 108)):
            surpass = True
    return surpass


def reverse_hand(data, bounds=False):
    list_notes, list_onsets, list_durations, list_fingers, list_ids, list_lengths, list_edges = [], [], [], [], [], [], []
    for notes, onsets, durations, fingers, ids, lengths, edges in zip(*data):
        new_notes = []
        notes = notes * 127
        jdx = 0
        for idx, n in enumerate(notes):
            if n == 0:
                jdx += 1
                new_notes.append(0)
            elif idx == jdx:
                new_notes.append(first_note_symmetric(notes[idx]))
            else:
                is_black_current = (n % 12) in [1, 3, 6, 8, 10]
                distance = n - notes[idx - 1]
                new_n = new_notes[-1] - distance
                is_black_new = (new_n % 12) in [1, 3, 6, 8, 10]
                new_notes.append(new_n)
                assert is_black_current == is_black_new, " is not working symmetric hand data augmentation " \
                                                         f"original seq = {np.array(notes)} " \
                                                         f"new seq = {np.array(new_notes)}"

        new_notes = np.array(new_notes)
        if bounds:
            if _surpass_bounds(new_notes):
                print(f"surpass piano keyboard bounds "
                      f"original seq = {np.array(notes)} "
                      f"new seq = {np.array(new_notes)}")
                continue
        list_notes.append(new_notes / 127)
        list_onsets.append(onsets)
        list_durations.append(durations)
        list_fingers.append(fingers)
        list_ids.append(ids)
        list_lengths.append(lengths)
        list_edges.append(edges)
    return list_notes, list_onsets, list_durations, list_fingers, list_ids, list_lengths, list_edges

def edges_to_matrix(edges, num_notes, graph_keys=GGCN.GRAPH_KEYS):
    if len(graph_keys) == 0:
        return None
    num_keywords = len(graph_keys)
    graph_dict = {key: i for i, key in enumerate(graph_keys)}
    if 'rest_as_forward' in graph_dict:
        graph_dict['rest_as_forward'] = graph_dict['forward']
        num_keywords -= 1
    matrix = np.zeros((num_keywords * 2, num_notes, num_notes))
    edg_indices = [(graph_dict[edg[2]], edg[0], edg[1])
                   for edg in edges
                   if edg[2] in graph_dict]
    reverse_indices = [(edg[0] + num_keywords, edg[2], edg[1]) if edg[0] != 0 else
                       (edg[0], edg[2], edg[1]) for edg in edg_indices]
    edg_indices = np.asarray(edg_indices + reverse_indices)

    matrix[edg_indices[:, 0], edg_indices[:, 1], edg_indices[:, 2]] = 1

    matrix[num_keywords, :, :] = np.identity(num_notes)
    matrix = torch.Tensor(matrix)
    return matrix


def predict_score(model, pitches, onsets, durations, hand):

    device = 'cpu' #'cuda:0' if torch.cuda.is_available() else 'cpu'
    #print("predict score")
    if hand == 'lh':
        data = [pitches], [onsets], [durations], [0], [0], [0], [0]
        data = reverse_hand(data)
        pitches = data[0][0]
    edges = compute_edge_list(onsets, pitches)
    edges = edges_to_matrix(edges, len(pitches))
    pitches = torch.Tensor(pitches).view(1, -1, 1)
    onsets = torch.Tensor(onsets).view(1, -1, 1)
    durations = torch.Tensor(durations).view(1, -1, 1)

    model.to(device)
    # print(len(pitches.shape))
    model.eval()
    with torch.no_grad():
        out, embedding = model.get_embedding(pitches.to(device),
                                             onsets.to(device),
                                             durations.to(device),
                                             torch.Tensor([pitches.shape[1]]).to(device),
                                             edges.to(device),
                                             None, beam_k=6)
    fingers_piece = [x + 1 for x in out[0]]
    if hand == 'lh':
        fingers_piece = [-1 * ff for ff in fingers_piece]
    return fingers_piece, embedding


def remove_fingerings(articulations):
    return [x for x in articulations if not x.isClassOrSubclass((music21.articulations.Fingering,))]

def compute_piece_argnn(m21stream):
    architecture = "ArGNNThumb-s"
    model = {
        'lh': choice_model('lh', architecture),
        'rh': choice_model('rh', architecture)
    }
    try:
        for hand in ['rh', 'lh']:
            om = utils.strm2map(m21stream, hand)
            onsets = np.array([o['offsetSeconds'] for o in om])
            pitches = np.array([((int(o['element'].pitch.midi) - 12) / 127.0) for o in om])

            fingers, embedding = predict_score(
                model[hand],
                pitches,
                onsets,
                [],
                hand,
            )

            for o in om:
                if 'chord' in o:
                    music21_structure = o['chord']
                else:
                    music21_structure = o['element']
                music21_structure.articulations = remove_fingerings(music21_structure.articulations)

            for o, finger in zip(om, fingers):
                if 'chord' in o:
                    music21_structure = o['chord']
                else:
                    music21_structure = o['element']
                f = music21.articulations.Fingering(finger)

                music21_structure.articulations = music21_structure.articulations + [f]
    except Exception as e:
        print("Finger detedction error")
        print(e)
        traceback.print_exc()
        return False
    return True

def get_args():
    """
    Parse arguments (if it's invoked as a command).
    """
    parser = argparse.ArgumentParser(
        description='Estimate fingering and store it to musicxml file.')
    parser.add_argument(
        '-a', '--annotation', type=str, help='input annotation file in .xml file', required=True)
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
        '-c', '--correct-only', action='store_true', default=False, help='Consider only proofread entries',
        required=False)
    args = parser.parse_args()
    infile = args.annotation
    base_path = args.base
    debug = args.debug
    correct_only = args.correct_only
    files_directory = args.files

    return infile, base_path, debug, correct_only, files_directory


if __name__ == '__main__':
    in_file, base_path, debug, correct_only, files_directory = get_args()
    songs = get_song_map(in_file, correct_only=correct_only).values()
    for s in songs:
        score_path = os.path.join(base_path, s['score'])
        print("Processing %s..." % score_path)
        path, name = os.path.split(score_path)
        out_xml_path = os.path.join(files_directory, name)
        compute_piece_argnn(score_path, out_xml_path, show=False)

