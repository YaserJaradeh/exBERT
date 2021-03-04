import os


def convert_ids_to_dict(data_dir: str, file_type: str):
    result = {}
    with open(os.path.join(data_dir, f"{file_type}2id.txt"), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            if index == 0 or len(line.strip()) == 0:
                continue
            parts = line.strip().split('\t')
            result[parts[0]] = parts[1]
    return result
