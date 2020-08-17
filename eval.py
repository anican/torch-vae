import argparse
import numpy as np
import os
import torch.backends.cudnn as cudnn
import yaml


from models import *
from vae_experiment import VariationalAutoencoderExperiment

def main():
    parser = argparse.ArgumentParser('Create embeddings from a trained model')
    parser.add_argument('--config', '-c', dest='filename', metavar='FILE',
            default='configs/vae.yaml')
    args = parser.parse_args()

    with open(args.filename, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    print(config)

    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False

    params = config['exp_params']
    # model = vae_models[config['model_params']['name']](**config['model_params'])
    PATH = '/homes/gws/anirudhc/torch-vae/logs/VanillaVAE/version_1/checkpoints/epoch=29.ckpt'
    experiment = VariationalAutoencoderExperiment.load_from_checkpoint(PATH, params=params)


if __name__ == '__main__':
    main()

