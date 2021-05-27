from pathlib import Path
import wget
import argparse
import zipfile

def retrieve_datasets(datasets:list, names:list, out_folder:Path) -> None:
    for url, name in zip(datasets, names):
        if name is not None:
            wget.download(url, str(out_folder / f'{name}.zip'))
        else:
            wget.download(url, str(out_folder))
        print()

def extract_datasets(out_folder:Path) -> None:
    zipfiles = sorted(out_folder.rglob("*.zip"))
    for path_to_zip_file in zipfiles:
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(out_folder / path_to_zip_file.stem)
        path_to_zip_file.unlink()

def parseArgumentsFromCommandLine():
    parser = argparse.ArgumentParser('uci data downloader script')
    parser.add_argument('-d', '--datasets', action='append', default=None)
    parser.add_argument('-n', '--names', action='append', default=[])
    parser.add_argument('-f', '--out_folder', type=Path, default='./data/')
    return parser.parse_args()

if __name__ == '__main__':
    args = parseArgumentsFromCommandLine()

    args.out_folder.mkdir(parents=True, exist_ok=True)
    if not args.out_folder.is_dir():
        raise ValueError('out_folder must be a folder, not a file')

    if args.datasets is not None:
        retrieve_datasets(args.datasets, args.names, args.out_folder)
    else:
        print('No datasets defined, append datasets with the -d option')

    extract_datasets(args.out_folder)
