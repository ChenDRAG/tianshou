import numpy as np
from torch.utils.tensorboard import SummaryWriter
def convert_tfevents_to_csv(dir = './', suffix = 'sorted'):
    """recursively find all tfevent file under dir, and create
    a csv file compatible with openai baseline. This function assumes that 
    there is at most one tfevents file in each directory and will add suffix
    to that directory.#TODO this need to be optimized later
    you can use 
    rl_plotter --save --avg_group --shaded_std --filename=test_rew --smooth=10
    to create standard rl reward graph.
    for more detail, please refer to https://github.com/gxywy/rl-plotter
    """
    import os
    from tensorboard.backend.event_processing import event_accumulator
    import csv
    import json
    if '-' in suffix:
        import warnings
        warnings.warn(
                    "suffix cannot contain character -."
                    "- in suffix has been replaced with _.",
                    Warning)
        suffix = suffix.replace('-', '_')
    dirs = []
    for dirname, _, files in os.walk(dir):
        for f in files:
            if 'tfevents' in f:
                print("handling " + os.path.join(dirname, f))
                ea=event_accumulator.EventAccumulator(os.path.join(dirname, f)) 
                ea.Reload()
                csv_file = open(os.path.join(dirname, 'test_rew.csv'), 'w')
                header={"t_start": ea._first_event_timestamp, 'env_id' : 'default'}
                header = '# {} \n'.format(json.dumps(header))
                csv_file.write(header)
                logger = csv.DictWriter(csv_file, fieldnames=('r', 'l', 't'))
                logger.writeheader()
                csv_file.flush()
                rews = ea.scalars.Items('test/rew')
                for rew in rews:
                    epinfo = {"r": rew.value, "l": rew.step, "t": rew.wall_time}
                    logger.writerow(epinfo)
                    csv_file.flush()
                csv_file.close()
                dirs.append(dirname)
                break
    for dirname in dirs:
        head, tail = os.path.split(dirname)
        os.rename(dirname, os.path.join(head, tail.replace('-', '_') + '-' + suffix))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='/home/huayu/git/tianshou/examples/mujoco/ppo_official3/Ant-v3/ppo')
    parser.add_argument('--suffix', type=str, default='Ant-v3_ppo_official3')
    args = parser.parse_args()
    convert_tfevents_to_csv(args.dir, args.suffix)
