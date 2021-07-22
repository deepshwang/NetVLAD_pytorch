import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='Argument Parser for NetVLAD')

    ##### Training Configurations #####
    parser.add_argument('--config_type',
                        type=str,
                        default='/media/TrainDataset/sungwon95/experiments/Lfin/NetVLAD_Tokyo/2021_07_14_01/config.yaml',
                        help='config path direction (modularized for every experiment)')



    args = parser.parse_args()

    return args