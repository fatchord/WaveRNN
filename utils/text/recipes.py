from utils.files import get_files


def ljspeech(path) :

    csv_file = get_files(path, extension='.csv')
    assert len(csv_file) == 1

    text_dict = {}

    with open(csv_file, encoding='utf-8') as f :
        for line in f :
            split = line.split('|')
            text_dict[split[0]] = split[-1]

    return text_dict