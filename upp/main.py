"""
Preprocessing pipeline for jet taggging.

By default all stages for the training split are run.
To run with only specific stages enabled, include the flag for the required stages.
To run without certain stages, include the corresponding negative flag.

Note that all stages are required to run the pipeline. If you want to disable resampling,
you need to set method: none in your config file.
"""
import argparse
from datetime import datetime
from pathlib import Path

from upp.classes.preprocessing_config import PreprocessingConfig
from upp.logger import setup_logger
from upp.stages import hist, plot
from upp.stages.merging import Merging
from upp.stages.normalisation import Normalisation
from upp.stages.resampling import Resampling


class HelpFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    ...


def parse_args():
    abool = argparse.BooleanOptionalAction
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=HelpFormatter,
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to config file")
    parser.add_argument("--prep", action=abool, help="Estimate and write PDFs")
    parser.add_argument("--resample", action=abool, help="Run resampling")
    parser.add_argument("--merge", action=abool, help="Run merging")
    parser.add_argument("--norm", action=abool, help="Compute normalisations")
    parser.add_argument("--plot", action=abool, help="Plot resampled distributions")
    splits = ["train", "val", "test", "all"]
    parser.add_argument("--split", default="train", choices=splits, help="Which file to produce")

    args = parser.parse_args()
    d = vars(args)
    ignore = ["config", "split"]
    if not any(v for a, v in d.items() if a not in ignore):
        for v in d:
            if v not in ignore and d[v] is None:
                d[v] = True
    return args


def run_pp(args) -> None:
    log = setup_logger()

    # print start info
    log.info("[bold green]Starting preprocessing...")
    start = datetime.now()
    log.info(f"Start time: {start.strftime('%Y-%m-%d %H:%M:%S')}")

    # load config
    config = PreprocessingConfig.from_file(Path(args.config), args.split)

    # create virtual datasets and pdf files
    if args.prep and args.split == "train":
        hist.main(config)

    # run the resampling
    if args.resample:
        resampling = Resampling(config)
        resampling.run()

    # run the merging
    if args.merge:
        merging = Merging(config)
        merging.run()

    # run the normalisation
    if args.norm and args.split == "train":
        norm = Normalisation(config)
        norm.run()

    # make plots
    if args.plot:
        plot.main(config, args.split)

    # print end info
    end = datetime.now()
    title = " Finished Preprocessing! "
    log.info(f"[bold green]{title:-^100}")
    log.info(f"End time: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Elapsed time: {str(end - start).split('.')[0]}")


def main() -> None:
    args = parse_args()
    log = setup_logger()

    if args.split == "all":
        d = vars(args)
        for split in ["train", "val", "test"]:
            d["split"] = split
            log.info(f"[bold blue]{'-'*100}")
            title = f" {args.split} "
            log.info(f"[bold blue]{title:-^100}")
            log.info(f"[bold blue]{'-'*100}")
            run_pp(args)
    else:
        run_pp(args)


if __name__ == "__main__":
    main()
