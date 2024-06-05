from tqdm import tqdm
import argparse
import wandb
from wandb.apis.importers.wandb import WandbImporter
from wandb.apis.importers import Namespace


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity', type = str)
    parser.add_argument('--project', type = str)
    parser.add_argument('--attribute', type = str)
    parser.add_argument('--delete-gt', type = str)
    return parser.parse_args()


def main():
    args = parse_args()

    runs = wandb.Api().runs(f"{args.entity}/{args.project}")
    
    for run in tqdm(runs):

        if run._attrs[args.attribute] > args.delete_gt:
            print(run.group, run.name)
            run.delete(delete_artifacts=True)

    
if __name__ == '__main__':
    main()