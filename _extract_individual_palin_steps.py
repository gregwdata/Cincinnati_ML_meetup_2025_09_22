import numpy as np
import json
import os

ref_sequence_filter = ['Palin']

def _nearest_idx(times: np.ndarray, t: float) -> int:
    # robust nearest by absolute difference
    return int(np.argmin(np.abs(times - t)))

def load_ref_sequences(reference_file: str) -> tuple[list[np.ndarray], list[str]]:
    """Load timeseries windows + labels from a Label Studio export JSON.

    Expected shape (simplified):
        data[0]['data']['ts']['acc']      -> list/array of values
        data[0]['data']['ts']['acc_time'] -> list/array of times (same length)
        data[0]['annotations'][0]['result'] -> regions with 'value': {'start', 'end', 'timeserieslabels': [label]}
    """
    with open(reference_file, "r") as f:
        ref_data = json.load(f)

    ref_sequence_filter_list = ref_sequence_filter

    sequences: list[np.ndarray] = []
    labels: list[str] = []
    for i in range(len(ref_data)):

        acc = np.asarray(ref_data[i]["data"]["ts"]["acc"], dtype=float)
        acc_time = np.asarray(ref_data[i]["data"]["ts"]["acc_time"], dtype=float)

        annotations = ref_data[i]["annotations"][0]["result"]
        for a in annotations:
            v = a["value"]
            t0 = float(v["start"])  # seconds
            t1 = float(v["end"])    # seconds
            label = str(v["timeserieslabels"][0])
            i0 = _nearest_idx(acc_time, t0)
            i1 = _nearest_idx(acc_time, t1)
            if i1 < i0:
                i0, i1 = i1, i0
            # include endpoint
            seq = acc[i0 : i1 + 1].astype(float, copy=True)
            if seq.size >= 2:
                if label in ref_sequence_filter_list or (len(ref_sequence_filter_list) == 0):
                    sequences.append(seq)
                    labels.append(label)
    if not sequences:
        raise ValueError("No reference sequences parsed from JSON; check your export format.")
    return sequences, labels


reference_files = [
    #'label_studio_export/project-2-at-2025-07-26-03-40-4a2f6183.json',
    #'label_studio_export/forward_walk_export.json',
    'label_studio_export/label_studio_export.json' # contains all steps
    ]

ref_sequences = []
ref_labels = []
for reference_file in reference_files:
    file_sequences, file_labels = load_ref_sequences(reference_file)
    ref_sequences.extend(file_sequences)
    ref_labels.extend(file_labels)

for seq, lab, i in zip(ref_sequences,ref_labels,range(len(ref_sequences))):
    # save the data to a file that can be imported to label studio with label forward step for each of the
    # identified motifs

    os.makedirs('./label_studio/isolated_Palin_steps', exist_ok=True)

    make_time = (0.01 * np.arange(len(seq))).tolist()

    # save a label-studio JSON file of the pre-annotated data
    preannotations = [
    {
        "data": {
            "ts": {
            "acc_time": make_time,
            "acc": seq.tolist()
            }
        } 
        ,
        "annotations": [
        ]
    }
    ]
    # save JSON to file
    json_file = f'./label_studio/isolated_Palin_steps/palin_step_{i}.json'
    with open(json_file, 'w') as f:
        json.dump(preannotations, f)
