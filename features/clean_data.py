import json


def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

basic_features = [
  'rh_pitch_set_lz',
  'lh_pitch_set_lz',
  'rh_pitch_range',
  'lh_pitch_range',
  'lh_average_pitch',
  'rh_average_pitch',
  'lh_average_ioi_seconds',
  'rh_average_ioi_seconds',
  'rh_displacement_rate',
  'lh_displacement_rate',
  'lh_pitch_entropy',
  'rh_pitch_entropy'
]

def main():
    new_data = []
    for element in load_json('current_difficulties.json'):
        new_element = {}
        new_element['id'] = element['id']
        new_element['grade'] = element['grade']
        for i, feature in enumerate(basic_features):
            new_element[feature] = element[feature]

        new_data.append(new_element)
    save_json(new_data, 'cipi-features-ISMIR24.json')


if __name__ == '__main__':

    main()






