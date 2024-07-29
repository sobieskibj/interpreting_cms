from pathlib import Path
from omegaconf import DictConfig

def _extract_output_dir(config: DictConfig) -> Path:
    '''
    Extracts path to output directory created by Hydra as pathlib.Path instance
    '''
    date = '/'.join(list(config._metadata.resolver_cache['now'].values()))
    output_dir = Path.cwd() / 'outputs'
    return output_dir if config.exp.log_determ else output_dir / date

def preprocess_config(config: DictConfig) -> None:
    config.exp.log_dir = _extract_output_dir(config)