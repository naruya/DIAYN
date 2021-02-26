from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
from tqdm import tqdm
import uuid
from rlkit.core import logger

filename = str(uuid.uuid4())
import torch
from gym.envs.mujoco import HalfCheetahEnv
from rlkit.envs.wrappers import NormalizedBoxEnv

def simulate_policy(args):
    data = torch.load(str(args.file))
    #data = joblib.load(str(args.file))
    policy = data['evaluation/policy']
    env = NormalizedBoxEnv(HalfCheetahEnv())
    #env = data['evaluation/env']
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()

    if args.collect:
        data = []
    for trial in tqdm(range(100)):
        path = rollout(
            env,
            policy,
            max_path_length=args.H+1,
            render=not args.collect,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()
        if args.collect:
            data.append([path['actions'], path['next_observations']])

    if args.collect:
        import pickle
        with open("data/expert.pkl", mode='wb') as f:
            pickle.dump(data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--collect', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
