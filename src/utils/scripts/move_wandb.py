import argparse

from wandb.apis.importers.wandb import WandbImporter
from wandb.apis.importers import Namespace


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-entity', type = str)
    parser.add_argument('--from-project', type = str)
    parser.add_argument('--from-api-key', type = str)
    parser.add_argument('--to-entity', type = str)
    parser.add_argument('--to-project', type = str)
    parser.add_argument('--to-api-key', type = str)
    return parser.parse_args()


def main():
    args = parse_args()

    importer = WandbImporter(
        src_base_url = "https://wandb.ai",
        src_api_key = args.from_api_key,
        dst_base_url = "https://wandb.ai/home",
        dst_api_key = args.to_api_key,
    )

    namespaces = [
        Namespace(args.from_entity, args.from_project)]

    remapping = {
        Namespace(args.from_entity, args.from_project): Namespace(args.to_entity, args.to_project)}

    importer.import_all(
        namespaces = namespaces, 
        remapping = remapping)
    
    
if __name__ == '__main__':
    main()