import threading
from torch.utils import tensorboard
from typing import Any, Dict, Optional


class SummaryWriter(tensorboard.SummaryWriter):
    """A more convenient Summary Writer(`tensorboard.SummaryWriter`).

    You can get the same instance of summary self.writer everywhere after you
    created one.
    ::

        >>> writer1 = SummaryWriter.get_instance(
                key="first", log_dir="log/test_sw/first")
        >>> writer2 = SummaryWriter.get_instance()
        >>> writer1 is writer2
        True
        >>> writer4 = SummaryWriter.get_instance(
                key="second", log_dir="log/test_sw/second")
        >>> writer5 = SummaryWriter.get_instance(key="second")
        >>> writer1 is not writer4
        True
        >>> writer4 is writer5
        True
    """

    _mutex_lock = threading.Lock()
    _default_key: str
    _instance: Optional[Dict[str, "SummaryWriter"]] = None

    @classmethod
    def get_instance(
        cls,
        key: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> "SummaryWriter":
        """Get instance of torch.utils.tensorboard.SummaryWriter by key."""
        with SummaryWriter._mutex_lock:
            if key is None:
                key = SummaryWriter._default_key
            if SummaryWriter._instance is None:
                SummaryWriter._instance = {}
                SummaryWriter._default_key = key
            if key not in SummaryWriter._instance.keys():
                SummaryWriter._instance[key] = SummaryWriter(*args, **kwargs)
        return SummaryWriter._instance[key]

class LazyLogger:
    def log_trainingdata(self):
        pass
    def log_updatedata(self, update_result, env_step, gradient_step):
        pass
    def log_testdata(self, collect_result, env_step, gradient_step):
        pass
    def global_log(self):
        pass

class DefaultStepLogger:
    def __init__(self, writer, log_train_interval = 1, log_update_interval = 1000):
        self.writer = writer
        self.n_trainlog = 0
        self.n_testlog = 0
        self.n_updatelog = 0
        self.log_train_interval = log_train_interval
        self.log_update_interval = log_update_interval
        self.log_train_count = 0
        self.log_update_count = 0
        
    def write(self, key, x, y):
        self.writer.add_scalar(key, y, global_step=x)

    def log_traindata(self, collect_result, env_step, gradient_step):
        if collect_result["n/ep"] > 0:
            if self.log_train_count % self.log_train_interval == 0:
                self.n_trainlog += 1
                if 'rew' not in collect_result:
                    collect_result['rew'] = collect_result['rews'].mean()
                if 'len' not in collect_result:
                    collect_result['len'] = collect_result['lens'].mean()
                self.write("train/n/ep", env_step, collect_result["n/ep"])  
                self.write("train/rew", env_step, collect_result["rew"])
                self.write("train/len", env_step, collect_result["len"])
            self.log_train_count += 1

    def log_testdata(self, collect_result, env_step, gradient_step):
        assert(collect_result["n/ep"] > 0)
        self.n_testlog += 1
        if 'rew' not in collect_result:
            collect_result['rew'] = collect_result['rews'].mean()
        if 'len' not in collect_result:
            collect_result['len'] = collect_result['lens'].mean()
        if 'rew_std' not in collect_result:
            collect_result['rew_std'] = collect_result['rews'].std()
        if 'len_std' not in collect_result:
            collect_result['len_std'] = collect_result['lens'].std()  
        self.write("test/rew", env_step, collect_result["rew"]) 
        self.write("test/len", env_step, collect_result["len"]) 
        self.write("test/rew_std", env_step, collect_result["rew_std"]) 
        self.write("test/len_std", env_step, collect_result["len_std"])             

    def log_updatedata(self, update_result, env_step, gradient_step):
        if self.log_update_count % self.log_update_interval == 0:
            self.n_updatelog += 1
            for k,v in update_result.items():
                self.write(k, gradient_step, v)
        self.log_update_count += 1

    def global_log(self):
        pass


def convert_tfevents_to_csv(dir = './', suffix = 'sorted'):
    """recursively find all tfevent file under dir, and create
    a csv file compatible with openai baseline. This function assumes that 
    there is at most one tfevents file in each directory and will add suffix
    to that directory.
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
    parser.add_argument('--dir', type=str, default='/home/huayu/git/tianshou/examples/mujoco/ddpgbenchmark/InvertedDoublePendulum-v2/ddpg')
    parser.add_argument('--suffix', type=str, default='InvertedDoublePendulum-v2_ddpg')
    args = parser.parse_args()
    convert_tfevents_to_csv(args.dir, args.suffix)
