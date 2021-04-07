import collections
import copy
import functools
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import jacinle.random as random
import jacinle.io as io

from difflogic.cli import format_args
from difflogic.dataset.utils import ValidActionDataset
from difflogic.envs.blocksworld import make as make_env
from difflogic.nn.baselines import MemoryNet
from difflogic.nn.neural_logic import InputTransform, LogicInference, LogicMachine, LogitsInference
from difflogic.nn.neural_logic.modules._utils import meshgrid_exclude_self
from difflogic.nn.dlm.layer import DifferentiableLogicMachine
from difflogic.nn.dlm.neural_logic import DLMInferenceBase
from difflogic.nn.rl.reinforce import REINFORCELoss, REINFORCELogLoss
from difflogic.train import MiningTrainerBase
from difflogic.train.accum_grad import AccumGrad
from difflogic.myutils import RunningMeanStd

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger
from jacinle.logging import set_output_file
from jacinle.utils.container import GView
from jacinle.utils.meter import GroupMeters
from jactorch.optim.quickaccess import get_optimizer
from jactorch.utils.meta import as_cuda
from jactorch.utils.meta import as_numpy
from jactorch.utils.meta import as_tensor
from difflogic.tqdm_utils import tqdm_for

TASKS = ['final', 'stack', 'nlrl-Stack', 'nlrl-Unstack', 'nlrl-On', 'sort', 'path']

parser = JacArgumentParser()

parser.add_argument(
    '--model',
    default='dlm',
    choices=['nlm', 'memnet', 'dlm'],
    help='model choices, nlm: Neural Logic Machine, memnet: Memory Networks, dlm: Differentiable Logic Machine')

# NLM parameters, works when model is 'nlm'.
nlm_group = parser.add_argument_group('Neural Logic Machines')
DifferentiableLogicMachine.make_nlm_parser(
    nlm_group, {
        'depth': 7,
        'breadth': 3,
        'exclude_self': True,
        'logic_hidden_dim': []
    },
    prefix='nlm')
nlm_group.add_argument(
    '--nlm-attributes',
    type=int,
    default=8,
    metavar='N',
    help='number of output attributes in each group of each layer of the LogicMachine'
)

# MemNN parameters, works when model is 'memnet'.
memnet_group = parser.add_argument_group('Memory Networks')
MemoryNet.make_memnet_parser(memnet_group, {}, prefix='memnet')

parser.add_argument(
    '--task', required=True, choices=TASKS, help='tasks choices')

method_group = parser.add_argument_group('Method')
method_group.add_argument(
    '--concat-worlds',
    type='bool',
    default=True,
    metavar='B',
    help='concat the features of objects of same id among two worlds accordingly'
)
method_group.add_argument(
    '--pred-depth',
    type=int,
    default=None,
    metavar='N',
    help='the depth of nlm used for prediction task')
method_group.add_argument(
    '--pred-weight',
    type=float,
    default=0.1,
    metavar='F',
    help='the linear scaling factor for prediction task')


data_gen_group = parser.add_argument_group('Data Generation')
data_gen_group.add_argument(
    '--gen-method',
    default='dnc',
    choices=['dnc', 'edge'],
    help='method use to generate random graph')
data_gen_group.add_argument(
    '--gen-graph-pmin',
    type=float,
    default=0.3,
    metavar='F',
    help='control parameter p reflecting the graph sparsity')
data_gen_group.add_argument(
    '--gen-graph-pmax',
    type=float,
    default=0.3,
    metavar='F',
    help='control parameter p reflecting the graph sparsity')
data_gen_group.add_argument(
    '--gen-max-len',
    type=int,
    default=4,
    metavar='N',
    help='maximum length of shortest path during training')
data_gen_group.add_argument(
    '--gen-test-len',
    type=int,
    default=4,
    metavar='N',
    help='length of shortest path during testing')
data_gen_group.add_argument(
    '--gen-directed', action='store_true', help='directed graph')


MiningTrainerBase.make_trainer_parser(
    parser, {
        'epochs': 500,
        'epoch_size': 100,
        'test_epoch_size': 50,
        'test_number_begin': 10,
        'test_number_step': 10,
        'test_number_end': 50,
        'curriculum_start': 2,
        'curriculum_step': 1,
        'curriculum_graduate': 12,
        'curriculum_thresh_relax': 0.01,
        'curriculum_thresh': 1,
        'sample_array_capacity': 3,
        'enable_mining': True,
        'mining_interval': 10,
        'mining_epoch_size': 200,
        'mining_dataset_size': 200,
        'inherit_neg_data': True,
        'prob_pos_data': 0.6
    })

train_group = parser.add_argument_group('Train')
train_group.add_argument('--seed', type=int, default=None, metavar='SEED')
train_group.add_argument(
    '--use-gpu', action='store_true', help='use GPU or not')
train_group.add_argument(
    '--optimizer',
    default='AdamW',
    choices=['SGD', 'Adam', 'AdamW'],
    help='optimizer choices')
train_group.add_argument(
    '--lr',
    type=float,
    default=0.005,
    metavar='F',
    help='initial learning rate')
train_group.add_argument(
    '--lr-decay',
    type=float,
    default=0.9,
    metavar='F',
    help='exponential decay of learning rate per lesson')
train_group.add_argument(
    '--accum-grad',
    type=int,
    default=1,
    metavar='N',
    help='accumulated gradient (default: 1)')
train_group.add_argument(
    '--ntrajectory',
    type=int,
    default=1,
    metavar='N',
    help='number of trajectories to compute gradient')
train_group.add_argument(
    '--batch-size',
    type=int,
    default=4,
    metavar='N',
    help='batch size for extra prediction')
train_group.add_argument(
    '--candidate-relax',
    type=int,
    default=0,
    metavar='N',
    help='number of thresh relaxation for candidate')
train_group.add_argument(
    '--extract-path', action='store_true', help='extract path or not')
train_group.add_argument(
    '--extract-rule', action='store_true', help='extract rule')
train_group.add_argument(
    '--gumbel-noise-begin',
    type=float,
    default=0.1)
train_group.add_argument(
    '--dropout-prob-begin',
    type=float,
    default=0.001)
train_group.add_argument(
    '--tau-begin',
    type=float,
    default=1)
train_group.add_argument(
    '--last-tau',
    type=float,
    default=0.01)
train_group.add_argument(
    '--norm-rewards',
    type='bool',
    default=False)



rl_group = parser.add_argument_group('Reinforcement Learning')
rl_group.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    metavar='F',
    help='discount factor for accumulated reward function in reinforcement learning'
)
rl_group.add_argument(
    '--penalty',
    type=float,
    default=-0.01,
    metavar='F',
    help='a small penalty each step')
rl_group.add_argument(
    '--entropy-beta',
    type=float,
    default=0.2,
    metavar='F',
    help='entropy loss scaling factor')
rl_group.add_argument(
    '--entropy-beta-decay',
    type=float,
    default=0.8,
    metavar='F',
    help='entropy beta exponential decay factor')
rl_group.add_argument(
    '--dlm-noise',
    type=int,
    default=2,
    metavar='N',
    help='dlm noise handling')
rl_group.add_argument(
    '--reinforce-log',
    type='bool',
    default=False)
rl_group.add_argument(
    '--distribution',
    type=int,
    default=1, #0 NLRL, 1 softmax, 2 move e^F
    metavar='N',
    help='distribution used to transform reasonning to action selection')
rl_group.add_argument(
    '--no-decay',
    type='bool',
    default=False)

io_group = parser.add_argument_group('Input/Output')
io_group.add_argument(
    '--dump-dir', default=None, metavar='DIR', help='dump dir')
io_group.add_argument(
    '--dump-play',
    action='store_true',
    help='dump the trajectory of the plays for visualization')
io_group.add_argument(
    '--dump-fail-only', action='store_true', help='dump failure cases only')
io_group.add_argument(
    '--load-checkpoint',
    default=None,
    metavar='FILE',
    help='load parameters from checkpoint')

schedule_group = parser.add_argument_group('Schedule')
schedule_group.add_argument(
    '--runs', type=int, default=1, metavar='N', help='number of runs')
schedule_group.add_argument(
    '--early-drop-epochs',
    type=int,
    default=50,
    metavar='N',
    help='epochs could spend for each lesson, early drop')
schedule_group.add_argument(
    '--save-interval',
    type=int,
    default=10,
    metavar='N',
    help='the interval(number of epochs) to save checkpoint')
schedule_group.add_argument(
    '--test-interval',
    type=int,
    default=None,
    metavar='N',
    help='the interval(number of epochs) to do test')
schedule_group.add_argument(
    '--test-only', action='store_true', help='test-only mode')
schedule_group.add_argument(
    '--test-not-graduated',
    action='store_true',
    help='test not graduated models also')

args = parser.parse_args()

args.use_gpu = args.use_gpu and torch.cuda.is_available()
args.dump_play = args.dump_play and (args.dump_dir is not None)
args.epoch_size = args.epoch_size // args.ntrajectory

if args.dump_dir is not None:
    io.mkdir(args.dump_dir)
    args.log_file = os.path.join(args.dump_dir, 'log.log')
    set_output_file(args.log_file)
else:
    args.checkpoints_dir = None
    args.summary_file = None

if args.seed is not None:
    random.reset_global_seed(args.seed)

if 'nlrl' in args.task:
  args.concat_worlds = False
  args.penalty = None
  args.pred_weight = 0.0

  # no curriculum learning for NLRL tasks
  args.curriculum_start = 4
  args.curriculum_graduate = 4
  args.mining_epoch_size = 20

  # not used
  args.test_number_begin = 4
  args.test_number_step = 1
  args.test_number_end = 4
elif args.task in ['sort', 'path']:
    args.concat_worlds = False
    args.pred_weight = 0.0
#    args.curriculum_start = 3

if args.task == 'sort':
    from difflogic.envs.algorithmic import make as make_env
elif args.task == 'path':
    from difflogic.envs.graph import make as make_env
    make_env = functools.partial(
      make_env,
      pmin=args.gen_graph_pmin,
      pmax=args.gen_graph_pmax,
      directed=args.gen_directed,
      gen_method=args.gen_method)
else:
    from difflogic.envs.blocksworld import make as make_env
    make_env = functools.partial(make_env, random_order=True, exclude_self=True, fix_ground=True)

logger = get_logger(__file__)

class Model(nn.Module):
    """The model for blocks world tasks."""

    def __init__(self):
        super().__init__()
        self.transform = InputTransform('cmp', exclude_self=False)

        input_dims = None
        # The 4 dimensions are: world_id, block_id, coord_x, coord_y
        if args.task == 'final':
            input_dim = 4
            # current_dim = 4 * 3 = 12
            current_dim = transformed_dim = self.transform.get_output_dim(input_dim)
            self.feature_axis = 1 if args.concat_worlds else 2
        elif args.task == 'stack':
            input_dim = 2
            current_dim = transformed_dim = self.transform.get_output_dim(input_dim)
            self.feature_axis = 2
        elif args.task == 'sort':
            self.feature_axis = 2
            current_dim = transformed_dim = 6
        elif args.task == 'path':
            self.feature_axis = 1
            input_dims = [0 for _ in range(args.nlm_breadth + 1)]
            input_dims[1] = 2
            input_dims[2] = 2
            transformed_dim = [0, 2, 2]
        elif 'nlrl' in args.task:
            self.feature_axis = 2
            input_dims = [0 for _ in range(args.nlm_breadth + 1)]
            input_dims[1] = 2 # unary: isFloor & top
            if args.task == 'nlrl-On':
                input_dims[2] = 2 # binary: goal_on & on
                transformed_dim = [0, 2, 2]
            else:
                input_dims[2] = 1 # binary: on
                transformed_dim = [0, 2, 1]
        else:
            raise ()

        if args.model in ['dlm', 'nlm'] and input_dims is None:
            input_dims = [0 for _ in range(args.nlm_breadth + 1)]
            input_dims[2] = current_dim

        if args.model == 'memnet':
            self.feature = MemoryNet.from_args(current_dim, self.feature_axis, args, prefix='memnet')
            current_dim = self.feature.get_output_dim()
        elif args.model == 'nlm':
            self.features = LogicMachine.from_args(input_dims, args.nlm_attributes, args, prefix='nlm')
            current_dim = self.features.output_dims[self.feature_axis]
        elif args.model == 'dlm':
            self.features = DifferentiableLogicMachine.from_args(input_dims, args.nlm_attributes, args, prefix='nlm')
            current_dim = self.features.output_dims[self.feature_axis]
        else:
            raise ()

        self.final_transform = InputTransform('concat', exclude_self=False)
        if args.task == 'final':
            if args.concat_worlds:
                current_dim = (self.final_transform.get_output_dim(current_dim) + transformed_dim) * 2

        if args.model == 'dlm':
            self.pred_valid = DLMInferenceBase(current_dim, 1, False, 'root_valid')
            self.pred = DLMInferenceBase(current_dim, 1, False, 'root')
            if args.distribution == 2:
                self.ac_selector = ActionSelector(current_dim)

            self.tau = args.tau_begin
            self.dropout_prob = args.dropout_prob_begin
            self.gumbel_prob = args.gumbel_noise_begin

            self.update_stoch()
        else: # args.model == 'nlm'
            self.pred_valid = LogicInference(current_dim, 1, [])
            self.pred = LogitsInference(current_dim, 1, [])
        if args.reinforce_log:
            self.loss = REINFORCELogLoss()
        else:
            self.loss = REINFORCELoss()
        self.pred_loss = nn.BCELoss()
        self.force_decay = False
        self.rnorm = RunningMeanStd(shape=1)

    def update_stoch(self):
        if args.model == 'dlm':
            self.features.update_tau(self.tau)
            self.pred_valid.update_tau(self.tau)
            self.pred.update_tau(self.tau)

            self.features.update_gumbel_noise(self.gumbel_prob)
            self.pred_valid.update_gumbel_noise(self.gumbel_prob)
            self.pred.update_gumbel_noise(self.gumbel_prob)

            self.features.update_dropout_prob(self.dropout_prob)
            self.pred_valid.update_dropout_prob(self.dropout_prob)
            self.pred.update_dropout_prob(self.dropout_prob)

    def lowernoise(self):
        if args.model == 'dlm':
            self.pred.independant_noise_per_sample = False
            self.pred.with_gumbel = False
            self.pred.with_dropout = False

            self.pred_valid.independant_noise_per_sample = False
            self.pred_valid.with_gumbel = False
            self.pred_valid.with_dropout = False

            self.features.independant_noise_per_sample(False)
            self.features.with_gumbel(False)
            self.features.with_dropout(False)

    def restorenoise(self):
        if args.model == 'dlm':
            self.pred.independant_noise_per_sample = True
            self.pred.with_gumbel = True
            self.pred.with_dropout = True

            self.pred_valid.independant_noise_per_sample = True
            self.pred_valid.with_gumbel = True
            self.pred_valid.with_dropout = True

            self.features.independant_noise_per_sample(True)
            self.features.with_gumbel(True)
            self.features.with_dropout(True)

    def stoch_decay(self, lesson, train_succ):
        if (args.model == 'dlm' and not args.no_decay and lesson == args.curriculum_graduate and train_succ > 0.95) or self.force_decay:
            self.force_decay = True
            self.tau = self.tau * 0.995
            self.gumbel_prob = self.gumbel_prob * 0.98
            self.dropout_prob = self.dropout_prob * 0.98
            args.pred_weight = args.pred_weight * 0.98

            #considered it failed
            if self.tau <= 0.45:
                self.tau = args.tau_begin
                self.dropout_prob = args.dropout_prob_begin
                self.gumbel_prob = args.gumbel_noise_begin

            self.update_stoch()

    def forward(self, feed_dict):
        feed_dict = GView(feed_dict)

        if args.task in ['final', 'stack', 'sort']:
            states = feed_dict.states.float()
            batch_size = states.size(0)
        else:
            states = feed_dict.states
            batch_size = states[0].size(0)

        f, more_info = self.get_binary_relations(states)
        saved_for_fa = f
        if args.model == 'dlm':
            f = self.pred(f)
            logits = f[0].squeeze(dim=-1).view(batch_size, -1)
            logits = 1e-5 + logits * (1.0 - 2e-5)
            if args.distribution == 0:
                sigma = logits.sum(-1).unsqueeze(-1)
                policy = torch.where(sigma > 1.0, logits/sigma, logits + (1-sigma)/logits.shape[1])
            elif args.distribution == 1:
                policy = F.softmax(logits / args.last_tau, dim=-1).clamp(min=1e-20)
            elif args.distribution == 2:
                if self.training:
                    fa = self.ac_selector(saved_for_fa.detach())
                    policy = (fa.sigmoid() + 1e-5 )*logits
                else:
                    policy = logits
                policy = policy / policy.sum(-1).unsqueeze(-1)
            else:
                raise()

            if feed_dict.training:
                if 'saturation' in more_info.keys():
                    more_info['saturation'].extend(f[1]['saturation'])
                else:
                    more_info['saturation']=[f[1]['saturation']]

                if 'entropies' in more_info.keys():
                    more_info['entropies'].extend(f[1]['entropies'])
        else:
            logits = self.pred(f).squeeze(dim=-1).view(batch_size, -1)
            policy = F.softmax(logits, dim=-1).clamp(min=1e-20)

        if not feed_dict.training:
            return dict(policy=policy, logits=logits)

        loss, monitors = self.loss(policy, feed_dict.actions, feed_dict.discount_rewards, feed_dict.entropy_beta)
 
        if args.pred_weight != 0.0:
            pred_states = feed_dict.pred_states.float()
            f, _ = self.get_binary_relations(pred_states, depth=args.pred_depth)
            if args.model == 'dlm':
                f = self.pred_valid(f)[0].squeeze(dim=-1).view(pred_states.size(0), -1)
            else:
                f = self.pred_valid(f).squeeze(dim=-1).view(pred_states.size(0), -1)
            # Set minimal value to avoid loss to be nan.
            valid = f[range(pred_states.size(0)), feed_dict.pred_actions].clamp(min=1e-20)
            pred_loss = self.pred_loss(valid, feed_dict.valid)
            monitors['pred/accuracy'] = feed_dict.valid.eq((valid > 0.5).float()).float().mean()
            loss = loss + args.pred_weight * pred_loss
        if args.model == 'dlm':
            pred = (logits.detach().cpu() > 0.5).float()
            sat = 1 - (logits.detach().cpu() - pred).abs()
            monitors.update({'saturation/min': np.array(sat.min())})
            monitors.update({'saturation/mean': np.array(sat.mean())})
            saturation_inside = torch.cat([a.flatten() for a in more_info['saturation']])
            monitors.update({'saturation-inside/min': np.array(saturation_inside.cpu().min())})
            monitors.update({'saturation-inside/mean': np.array(saturation_inside.cpu().mean())})
            monitors.update({'tau': np.array(self.tau)})
            monitors.update({'dropout_prob': np.array(self.dropout_prob)})
            monitors.update({'gumbel_prob': np.array(self.gumbel_prob)})

        return loss, monitors, dict()

    def get_binary_relations(self, states, depth=None):
        """get binary relations given states, up to certain depth."""
        more_info = None
        if args.task in ['final', 'stack']:
            # total = 2 * the number of objects in each world
            total = states.size()[1]
            f = self.transform(states)
        else:
            f = states

        if args.model == 'memnet':
            f = self.feature(f)
        else:
            inp = [None for i in range(args.nlm_breadth + 1)]
            if type(f) is not list:
                inp[2] = f
            else:
                inp[1] = f[0]
                inp[2] = f[1]
            if args.model == 'dlm':
                path = None
                if not self.training and args.extract_path:
                    for i in range(len(inp)):
                        if inp[i] is None:
                            continue
                        inp[i] = inp[i].bool()
                    path = self.pred.weight.argmax(-1)
                features = self.features(inp, depth=depth, pathFrompred=path, feature_axis=self.feature_axis, extract_rule=args.extract_rule)
                f = features[0][self.feature_axis]
                more_info = features[1]
            else:
                features = self.features(inp, depth=depth)
                f = features[self.feature_axis]

        if args.task == 'final':
            assert total % 2 == 0
            nr_objects = total // 2
            if args.concat_worlds:
                # To concat the properties of blocks with the same id in both world.
                f = torch.cat([f[:, :nr_objects], f[:, nr_objects:]], dim=-1)
                states = torch.cat([states[:, :nr_objects], states[:, nr_objects:]], dim=-1)
                transformed_input = self.transform(states)
                # And perform a 'concat' transform to binary representation (relations).
                f = torch.cat([self.final_transform(f), transformed_input], dim=-1)
            else:
                f = f[:, :nr_objects, :nr_objects].contiguous()
        elif args.task == 'stack' or 'nlrl' in args.task:
            nr_objects = total if args.task == 'stack' else states[0].size()[1]
            f = f[:, :nr_objects, :nr_objects].contiguous()
        elif args.task in ['sort', 'path']:
            pass
        else:
            raise ()

        if args.task != 'path':
            f = meshgrid_exclude_self(f)
        return f, more_info


def make_data(traj, gamma):
    """Aggregate data as a batch for RL optimization."""
    q = 0
    discount_rewards = []
    for reward in traj['rewards'][::-1]:
        q = q * gamma + reward
        discount_rewards.append(q)
    discount_rewards.reverse()

    if type(traj['states'][0]) is list:
        f1 = [f[0] for f in traj['states']]
        f2 = [f[1] for f in traj['states']]
        traj['states'] = [torch.cat(f1, dim=0), torch.cat(f2, dim=0)]
    else:
        traj['states'] = as_tensor(np.array(traj['states']))

    traj['actions'] = as_tensor(np.array(traj['actions']))
    traj['discount_rewards'] = as_tensor(np.array(discount_rewards)).float()
    return traj


def run_episode(env,
                model,
                mode,
                number,
                play_name='',
                dump=False,
                dataset=None,
                eval_only=False,
                use_argmax=False,
                need_restart=False,
                entropy_beta=0.0):
    """Run one episode using the model with $number blocks."""
    is_over = False
    traj = collections.defaultdict(list)
    score = 0
    if need_restart:
        env.restart()

    optimal = None
    if args.task == 'path':
        optimal = env.unwrapped.dist
        relation = env.unwrapped.graph.get_edges()
        relation = np.stack([relation, relation.T], axis=-1).astype(dtype=np.float32)
        st, ed = env.current_state
        nodes_trajectory = [int(st)]
        destination = int(ed)
        policies = []
    elif args.task == 'sort':
        optimal = env.unwrapped.optimal
        array = [str(i) for i in env.unwrapped.array]

    # If dump_play=True, store the states and actions in a json file
    # for visualization.
    dump_play = args.dump_play and dump
    if dump_play:
        nr_objects = number + 1
        array = env.unwrapped.current_state
        moves, new_pos, policies = [], [], []

    if args.model == 'dlm':
        # by default network isn't in training mode during data collection
        # but with dlm we don't want to use argmax only
        # except in 2 cases (testing the interpretability or the last mining phase to get an interpretable policy):
        if mode == 'test-inter' or (mode in ['mining', 'inherit'] and number == args.curriculum_graduate):
            model.lowernoise()
        else:
            model.train(True)

            if args.dlm_noise == 1 and mode in ['mining', 'inherit', 'test', 'test-inter']:
                model.lowernoise()
            elif args.dlm_noise == 2:
                model.lowernoise()

    step = 0
    while not is_over:
        if args.task == 'path':
            st, ed = env.current_state
            state = np.zeros((relation.shape[0], 2), dtype=np.float32)
            state[st, 0] = 1
            state[ed, 1] = 1
            feed_dict = dict(states=[np.array([state]), np.array([relation])])
        else:
            state = env.current_state
            if 'nlrl' not in args.task or args.task == 'sort':
                feed_dict = dict(states=np.array([state]))
            else:
                feed_dict = dict(states=state)
        feed_dict['entropy_beta'] = as_tensor(entropy_beta).float()
        feed_dict['training'] = as_tensor(False)
        feed_dict = as_tensor(feed_dict)

        with torch.set_grad_enabled(False):
            output_dict = model(feed_dict)
        policy = output_dict['policy']
        p = as_numpy(policy.data[0])
        action = p.argmax() if use_argmax else random.choice(len(p), p=p)
        if args.pred_weight != 0.0:
            # Need to ensure that the env.utils.MapActionProxy is the outermost class.
            mapped_x, mapped_y = env.mapping[action]
            # env.unwrapped to get the innermost Env class.
            valid = env.unwrapped.world.moveable(mapped_x, mapped_y)
        reward, is_over = env.action(action)
        step += 1
        if dump_play:
            moves.append([mapped_x, mapped_y])
            res = tuple(env.current_state[mapped_x][2:])
            new_pos.append((int(res[0]), int(res[1])))

            logits = as_numpy(output_dict['logits'].data[0])
            tops = np.argsort(p)[-10:][::-1]
            tops = list(
                map(lambda x: (env.mapping[x], float(p[x]), float(logits[x])), tops))
            policies.append(tops)
        # For now, assume reward=1 only when succeed, otherwise reward=0.
        # Manipulate the reward and get success information according to reward.
        if reward == 0 and args.penalty is not None:
            reward = args.penalty
        succ = 1 if is_over and reward > 0.99 else 0

        score += reward

        if type(feed_dict['states']) is list:
            traj['states'].append([f for f in feed_dict['states']])
        else:
            traj['states'].append(state)

        traj['rewards'].append(reward)
        traj['actions'].append(action)

        if args.pred_weight != 0.0:
            if not eval_only and dataset is not None and mapped_x != mapped_y:
                dataset.append(nr_objects, state, action, valid)

    # Dump json file as record of the playing.
    if dump_play and not (args.dump_fail_only and succ):
        array = array[:, 2:].astype('int32').tolist()
        array = [array[:nr_objects], array[nr_objects:]]
        json_str = json.dumps(
            # Let indent=True for an indented view of json files.
            dict(array=array, moves=moves, new_pos=new_pos,
                 policies=policies))
        dump_file = os.path.join(
            args.current_dump_dir,
            '{}_blocks{}.json'.format(play_name, env.unwrapped.nr_blocks))
        with open(dump_file, 'w') as f:
            f.write(json_str)

    length = step

    if args.model == 'dlm':
        model.restorenoise()

    return succ, score, traj, length, optimal


class MyTrainer(MiningTrainerBase):
    def save_checkpoint(self, name):
        if args.checkpoints_dir is not None:
            checkpoint_file = os.path.join(args.checkpoints_dir,
                                           'checkpoint_{}.pth'.format(name))
            super().save_checkpoint(checkpoint_file)

    def _dump_meters(self, meters, mode):
        if args.summary_file is not None:
            meters_kv = meters._canonize_values('avg')
            meters_kv['mode'] = mode
            meters_kv['epoch'] = self.current_epoch
            with open(args.summary_file, 'a') as f:
                f.write(io.dumps_json(meters_kv))
                f.write('\n')

    def _prepare_dataset(self, epoch_size, mode):
        pass

    def _get_player(self, number, mode, index):
        if args.task == 'path':
            test_len = args.gen_test_len
            test_mode = (args.model == 'dlm' and number == args.curriculum_graduate and mode == "mining") or ("test" in mode)
            dist_range = (test_len, test_len) if test_mode else (1, args.gen_max_len)
            player = make_env(args.task, number, dist_range=dist_range)
        elif 'nlrl' not in args.task or 'test' not in mode:
            player = make_env(args.task, number)
        else:
            #testing env. for NLRL
            #suppose 5 variations per env.
            player = make_env(args.task, number, variation_index=(index % 5))

        player.restart()
        return player

    def _get_result_given_player(self, index, meters, number, player, mode):
        assert mode in ['train', 'test', 'mining', 'inherit', 'test-inter', 'test-inter-deter', 'test-deter']
        params = dict(
            eval_only=True,
            number=number,
            play_name='{}_epoch{}_episode{}'.format(mode, self.current_epoch, index))
        backup = None
        if mode == 'train':
            params['eval_only'] = False
            params['dataset'] = self.valid_action_dataset
            params['entropy_beta'] = self.entropy_beta
            meters.update(lr=self.lr, entropy_beta=self.entropy_beta)
        elif 'test' in mode:
            params['dump'] = True
            params['use_argmax'] = 'deter' in mode
        else:
            backup = copy.deepcopy(player)
            params['use_argmax'] = index < (args.mining_epoch_size//2)

        if mode == 'train':
            if args.use_gpu:
                self.model.cpu()

            mergedfc = []
            for i in range(args.ntrajectory):
                succ, score, traj, length, optimal = run_episode(player, self.model, mode, need_restart=(i!=0), **params)
                if args.task in ['sort', 'path']:
                    meters.update(number=number, succ=succ, score=score, length=length, optimal=optimal)
                else:
                    meters.update(number=number, succ=succ, score=score, length=length)
                feed_dict = make_data(traj, args.gamma)
                # content from valid_move dataset
                if args.pred_weight != 0.0:
                    states, actions, labels = self.valid_action_dataset.sample_batch(args.batch_size)
                    feed_dict['pred_states'] = as_tensor(states)
                    feed_dict['pred_actions'] = as_tensor(actions)
                    feed_dict['valid'] = as_tensor(labels).float()
                mergedfc.append(feed_dict)

            for k in feed_dict.keys():
                if k not in ["rewards", "entropy_beta"]:  # reward not used to update loss
                    if type(mergedfc[0][k]) is list:
                        f1 = [j[k][0] for j in mergedfc]
                        f2 = [j[k][1] for j in mergedfc]
                        feed_dict[k] = [torch.cat(f1, dim=0), torch.cat(f2, dim=0)]
                    else:
                        feed_dict[k] = torch.cat([j[k] for j in mergedfc], dim=0)
            feed_dict['entropy_beta'] = as_tensor(self.entropy_beta).float()
            feed_dict['training'] = as_tensor(True)

            if args.norm_rewards:
                if args.accum_grad > 1:
                    feed_dict['discount_rewards'] = self.model.rnorm.obs_filter(feed_dict['discount_rewards'])
                elif feed_dict['discount_rewards'].shape[0] > 1:
                    feed_dict['discount_rewards'] = (feed_dict['discount_rewards'] - feed_dict['discount_rewards'].mean()) / (feed_dict['discount_rewards'].std() + 10 ** -7)

            #dirty trick
            if args.accum_grad > 1:
                self.optimizer.provide_batch_size(feed_dict['discount_rewards'].shape[0])

            if args.use_gpu:
                feed_dict = as_cuda(feed_dict)
                self.model.cuda()
            self.model.train()
            return feed_dict
        else:
            if args.use_gpu:
                self.model.cpu()
            succ, score, traj, length, optimal = run_episode(player, self.model, mode, **params)
            if args.task in ['sort', 'path']:
                meters.update(number=number, succ=succ, score=score, length=length, optimal=optimal)
                message = ('> {} iter={iter}, number={number}, succ={succ}, '
                       'score={score:.4f}, length={length}, optimal={optimal}').format(mode, iter=index, **meters.val)
            else:
                meters.update(number=number, succ=succ, score=score, length=length)
                message = ('> {} iter={iter}, number={number}, succ={succ}, '
                       'score={score:.4f}, length={length}').format(mode, iter=index, **meters.val)
            return message, dict(succ=succ, number=number, backup=backup)

    def _extract_info(self, extra):
        return extra['succ'], extra['number'], extra['backup']

    def _get_accuracy(self, meters):
        return meters.avg['succ']

    def _get_threshold(self):
        candidate_relax = 0 if self.is_candidate else args.candidate_relax
        return super()._get_threshold() - \
               self.curriculum_thresh_relax * candidate_relax

    def _upgrade_lesson(self):
        super()._upgrade_lesson()
        # Adjust lr & entropy_beta w.r.t different lesson progressively.
        self.lr *= args.lr_decay
        self.entropy_beta *= args.entropy_beta_decay
        self.set_learning_rate(self.lr)

    def _train_epoch(self, epoch_size):
        meters = super()._train_epoch(epoch_size)
        self.model.stoch_decay(self.current_number, meters.avg['succ'])

        i = self.current_epoch
        if args.save_interval is not None and i % args.save_interval == 0:
            self.save_checkpoint(str(i))
        if args.test_interval is not None and i % args.test_interval == 0:
            self.test()

        return meters

    def _early_stop(self, meters):
        t = args.early_drop_epochs
        if t is not None and self.current_epoch > t * (self.nr_upgrades + 1):
            return True
        return super()._early_stop(meters)

    def train(self):
        self.valid_action_dataset = ValidActionDataset()
        self.lr = args.lr
        self.entropy_beta = args.entropy_beta
        return super().train()

    def test(self):
        ret1 = super().test()
        ret2 = super().advanced_test(inter=False, deter=True)

        if args.model != 'dlm':
            return ret1 if ret1[-1].avg['score'] > ret2[-1].avg['score'] else ret2

        ret1 = super().advanced_test(inter=True, deter=False)
        ret2 = super().advanced_test(inter=True, deter=True)
        return ret1 if ret1[-1].avg['score'] > ret2[-1].avg['score'] else ret2


def main(run_id):
    if args.dump_dir is not None:
        if args.runs > 1:
            args.current_dump_dir = os.path.join(args.dump_dir,
                                                 'run_{}'.format(run_id))
            io.mkdir(args.current_dump_dir)
        else:
            args.current_dump_dir = args.dump_dir
        args.checkpoints_dir = os.path.join(args.current_dump_dir, 'checkpoints')
        io.mkdir(args.checkpoints_dir)
        args.summary_file = os.path.join(args.current_dump_dir, 'summary.json')

    logger.info(format_args(args))

    model = Model()
    optimizer = get_optimizer(args.optimizer, model, args.lr)
    if args.accum_grad > 1:
        optimizer = AccumGrad(optimizer, args.accum_grad)

    trainer = MyTrainer.from_args(model, optimizer, args)

    if args.load_checkpoint is not None:
        trainer.load_checkpoint(args.load_checkpoint)

    if args.test_only:
        trainer.current_epoch = 0
        return None, trainer.test()

    graduated = trainer.train()
    trainer.save_checkpoint('last')
    test_meters = trainer.test() if graduated or args.test_not_graduated else None
    return graduated, test_meters


if __name__ == '__main__':
    stats = []
    nr_graduated = 0

    for i in range(args.runs):
        graduated, test_meters = main(i)
        logger.info('run {}'.format(i + 1))

        if test_meters is not None:
            for j, meters in enumerate(test_meters):
                if len(stats) <= j:
                    stats.append(GroupMeters())
                stats[j].update(number=meters.avg['number'], test_succ=meters.avg['succ'])

            for meters in stats:
                logger.info('number {}, test_succ {}'.format(meters.avg['number'], meters.avg['test_succ']))

        if not args.test_only:
            nr_graduated += int(graduated)
            logger.info('graduate_ratio {}'.format(nr_graduated / (i + 1)))
            if graduated:
                for j, meters in enumerate(test_meters):
                    stats[j].update(grad_test_succ=meters.avg['succ'])
            if nr_graduated > 0:
                for meters in stats:
                    logger.info('number {}, grad_test_succ {}'.format(meters.avg['number'], meters.avg['grad_test_succ']))
