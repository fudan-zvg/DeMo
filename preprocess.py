from argparse import ArgumentParser
from pathlib import Path
from typing import List
import multiprocessing
from tqdm import tqdm
from src.datamodule.av2_extractor import Av2Extractor


def glob_files(data_root: Path, mode: str):
    file_root = data_root / mode
    scenario_files = list(file_root.rglob("*.parquet"))
    return scenario_files


def preprocess(args):
    batch = args.batch
    data_root = Path(args.data_root)

    for mode in ["train", "val", "test"]:
        save_dir = Path("data/DeMo_processed") / mode
        extractor = Av2Extractor(save_path=save_dir, mode=mode)

        save_dir.mkdir(exist_ok=True, parents=True)
        scenario_files = glob_files(data_root, mode)

        if args.parallel:
            with multiprocessing.Pool(16) as p:
                all_name = list(tqdm(p.imap(extractor.save, scenario_files), total=len(scenario_files)))
        else:
            for file in tqdm(scenario_files):
                extractor.save(file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", "-d", type=str, default='/path/to/data_root')
    parser.add_argument("--batch", "-b", type=int, default=50)
    parser.add_argument("--parallel", "-p", action="store_true")

    args = parser.parse_args()
    preprocess(args)
