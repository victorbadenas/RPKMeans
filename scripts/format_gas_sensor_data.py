from pathlib import Path
import argparse
import re
import tqdm

def iter_file(path:Path) -> str:
    with open(path, 'r') as f:
        for line in f:
            yield line.strip()

class CsvFileWriter:
    def __init__(self, path, headers=None, sep=','):
        self.path = Path(path)
        self.sep = sep
        self._create_file(headers)
        self.handler = open(self.path, 'a+')

    def _create_file(self, headers=None):
        if headers is not None:
            if not isinstance(headers, str):
                try:
                    headers = self.sep.join(headers)
                except Exception as e:
                    print('headers is not iterable')
                    raise e

            with open(self.path, 'w') as f:
                f.write(headers + '\n')
        else:
            self.path.touch()

    def append(self, line):
        if not isinstance(line, str):
            line = self.sep.join(line)
        self.handler.write(line + '\n')

    def close(self):
        self.handler.close()


def parseArgumentsFromCommandLine():
    parser = argparse.ArgumentParser('uci data downloader script')
    parser.add_argument('-f', '--data_folder', type=Path, default='./data/')
    return parser.parse_args()

if __name__ == '__main__':
    args = parseArgumentsFromCommandLine()
    data_folder = args.data_folder

    for file_path in data_folder.rglob('*.txt'):
        first = True
        for line in tqdm.tqdm(iter_file(file_path)):
            if first:
                headers = list(map(lambda x: x.strip().replace(' ', '_'), line.split(',')))[:3]
                headers += list(map(lambda x: f'attr{x}', range(16)))
                csv_file_writer = CsvFileWriter(file_path.with_suffix('.csv'), 
                    headers=headers)
                first = False
                continue
            line = re.sub(' +', ' ', line).replace(' ', ',')
            csv_file_writer.append(line)
        csv_file_writer.close()
        del csv_file_writer
        file_path.unlink()
