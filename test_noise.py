#!/usr/bin/env python3
"""
Test script for GCBF+ with edge feature noise in double_integrator environment.
This script demonstrates how to test the GCBF+ algorithm with Gaussian noise
added to edge features to evaluate robustness.
"""

import argparse
import datetime
import functools as ft
import os
import pathlib
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax import lax
import numpy as np
import yaml

from gcbfplus.algo import GCBF, GCBFPlus, make_algo, CentralizedCBF, DecShareCBF
from gcbfplus.env import make_env
from gcbfplus.env.base import RolloutResult
from gcbfplus.trainer.utils import get_bb_cbf
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import jax_jit_np, tree_index, chunk_vmap, merge01, jax_vmap


def test_gcbf_plus_with_edge_noise(args):
    """Test GCBF+ with edge feature noise in double_integrator environment."""
    print(f"> Testing GCBF+ with edge noise: {args}")
    
    stamp_str = datetime.datetime.now().strftime("%m%d-%H%M")
    
    # Set up environment variables and seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)
    np.random.seed(args.seed)
    
    # Load config from trained model
    config = None
    if args.path is not None:
        with open(os.path.join(args.path, "config.yaml"), "r") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)
    
    # Create double integrator environment with edge noise
    num_agents = config.num_agents if config and args.num_agents is None else args.num_agents
    
    # Create environment first
    env = make_env(
        env_id="DoubleIntegrator",
        num_agents=num_agents,
        num_obs=args.obs,
        area_size=args.area_size,
        max_step=args.max_step,
        max_travel=args.max_travel,
    )
    
    # Then update parameters with edge noise
    env_params = {
        "edge_noise_std": args.edge_noise_std,  # Key parameter for edge noise
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.5],
        "n_obs": args.obs,
        "m": 0.1,
    }
    
    # Update environment parameters
    env._params.update(env_params)
    
    # Load GCBF+ algorithm
    if args.path is not None:
        path = args.path
        model_path = os.path.join(path, "models")
        if args.step is None:
            models = os.listdir(model_path)
            step = max([int(model) for model in models if model.isdigit()])
        else:
            step = args.step
        print(f"Loading model from step: {step}")
        
        algo = make_algo(
            algo=config.algo,
            env=env,
            node_dim=env.node_dim,
            edge_dim=env.edge_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            n_agents=env.num_agents,
            gnn_layers=config.gnn_layers,
            batch_size=config.batch_size,
            buffer_size=config.buffer_size,
            horizon=config.horizon,
            lr_actor=config.lr_actor,
            lr_cbf=config.lr_cbf,
            alpha=config.alpha,
            eps=0.02,
            inner_epoch=8,
            loss_action_coef=config.loss_action_coef,
            loss_unsafe_coef=config.loss_unsafe_coef,
            loss_safe_coef=config.loss_safe_coef,
            loss_h_dot_coef=config.loss_h_dot_coef,
            max_grad_norm=2.0,
            seed=config.seed,
        )
        algo.load(model_path, step)
        act_fn = jax.jit(algo.act)
    else:
        raise ValueError("Model path is required for GCBF+ testing")
    
    # Test keys
    test_key = jr.PRNGKey(args.seed)
    test_keys = jr.split(test_key, 1_000)[: args.epi]
    test_keys = test_keys[args.offset :]
    
    # Create rollout function that demonstrates noise by running two comparisons
    def rollout_with_noise(key):
        """Run standard rollout - noise effect is shown by parameter setting."""
        
        def policy_fn(graph_obs):
            return act_fn(graph_obs)
        
        return env.rollout_fn(policy_fn, args.max_step)(key)
    
    # Test setup
    rollout_fn = jax_jit_np(rollout_with_noise)
    is_unsafe_fn = jax_jit_np(jax_vmap(env.collision_mask))
    is_finish_fn = jax_jit_np(jax_vmap(env.finish_mask))
    
    # Run test episodes
    rewards = []
    costs = []
    rollouts = []
    is_unsafes = []
    is_finishes = []
    rates = []
    
    print(f"Testing with edge noise std: {args.edge_noise_std}")
    print(f"Number of agents: {num_agents}")
    print(f"Area size: {args.area_size}")
    print(f"Number of obstacles: {args.obs}")
    
    # Verify that noise is actually being applied
    print("\nVerifying noise application:")
    test_key = jr.PRNGKey(42)
    test_graph = env.reset(test_key)
    test_action = jnp.zeros((num_agents, env.action_dim))
    
    # Test with no noise
    old_noise_std = env._params.get("edge_noise_std", 0.0)
    env._params["edge_noise_std"] = 0.0
    graph_no_noise = env.forward_graph(test_graph, test_action, test_key)
    
    # Test with noise
    env._params["edge_noise_std"] = args.edge_noise_std
    graph_with_noise = env.forward_graph(test_graph, test_action, test_key)
    
    # Restore original noise level
    env._params["edge_noise_std"] = old_noise_std
    
    # Check if graphs are different
    edge_diff = jnp.abs(graph_no_noise.edges - graph_with_noise.edges).max()
    print(f"Max edge feature difference: {edge_diff:.6f}")
    print(f"Noise is {'ACTIVE' if edge_diff > 1e-6 else 'INACTIVE'}")
    
    print("=" * 50)
    
    for i_epi in range(args.epi):
        key_x0, _ = jr.split(test_keys[i_epi], 2)
        
        rollout: RolloutResult = rollout_fn(key_x0)
        is_unsafes.append(is_unsafe_fn(rollout.Tp1_graph))
        is_finishes.append(is_finish_fn(rollout.Tp1_graph))
        
        epi_reward = rollout.T_reward.sum()
        epi_cost = rollout.T_cost.sum()
        rewards.append(epi_reward)
        costs.append(epi_cost)
        rollouts.append(rollout)
        
        if len(is_unsafes) == 0:
            continue
            
        safe_rate = 1 - is_unsafes[-1].max(axis=0).mean()
        finish_rate = is_finishes[-1].max(axis=0).mean()
        success_rate = (
            (1 - is_unsafes[-1].max(axis=0)) * is_finishes[-1].max(axis=0)
        ).mean()
        
        print(
            f"Episode {i_epi+1:2d}: "
            f"reward={epi_reward:6.3f}, cost={epi_cost:6.3f}, "
            f"safe={safe_rate*100:5.1f}%, finish={finish_rate*100:5.1f}%, "
            f"success={success_rate*100:5.1f}%"
        )
        
        rates.append(np.array([safe_rate, finish_rate, success_rate]))
    
    # Calculate final statistics
    is_unsafe = np.max(np.stack(is_unsafes), axis=1)
    is_finish = np.max(np.stack(is_finishes), axis=1)
    
    safe_mean, safe_std = (1 - is_unsafe).mean(), (1 - is_unsafe).std()
    finish_mean, finish_std = is_finish.mean(), is_finish.std()
    success_mean, success_std = ((1 - is_unsafe) * is_finish).mean(), (
        (1 - is_unsafe) * is_finish
    ).std()
    
    print("=" * 50)
    print("FINAL RESULTS:")
    print(f"Reward: {np.mean(rewards):6.3f} ± {np.std(rewards):6.3f}")
    print(f"Cost:   {np.mean(costs):6.3f} ± {np.std(costs):6.3f}")
    print(f"Safe rate:    {safe_mean*100:5.1f}% ± {safe_std*100:5.1f}%")
    print(f"Finish rate:  {finish_mean*100:5.1f}% ± {finish_std*100:5.1f}%")
    print(f"Success rate: {success_mean*100:5.1f}% ± {success_std*100:5.1f}%")
    print("=" * 50)
    
    # Save results
    if args.log:
        log_dir = os.path.join(args.path, "edge_noise_tests")
        os.makedirs(log_dir, exist_ok=True)
        
        logfile = os.path.join(log_dir, f"edge_noise_test_{stamp_str}.csv")
        with open(logfile, "w") as f:
            f.write("edge_noise_std,num_agents,num_episodes,area_size,num_obstacles,")
            f.write("safe_mean,safe_std,finish_mean,finish_std,success_mean,success_std,")
            f.write("reward_mean,reward_std,cost_mean,cost_std\n")
            f.write(f"{args.edge_noise_std},{num_agents},{args.epi},{args.area_size},{args.obs},")
            f.write(f"{safe_mean*100:.3f},{safe_std*100:.3f},")
            f.write(f"{finish_mean*100:.3f},{finish_std*100:.3f},")
            f.write(f"{success_mean*100:.3f},{success_std*100:.3f},")
            f.write(f"{np.mean(rewards):.3f},{np.std(rewards):.3f},")
            f.write(f"{np.mean(costs):.3f},{np.std(costs):.3f}\n")
        
        print(f"Results saved to: {logfile}")
        
        # Save detailed episode results
        detail_file = os.path.join(log_dir, f"edge_noise_episodes_{stamp_str}.csv")
        with open(detail_file, "w") as f:
            f.write("episode,reward,cost,safe_rate,finish_rate,success_rate\n")
            for i, (reward, cost, rate) in enumerate(zip(rewards, costs, rates)):
                f.write(f"{i+1},{reward:.3f},{cost:.3f},{rate[0]*100:.3f},{rate[1]*100:.3f},{rate[2]*100:.3f}\n")
        
        print(f"Episode details saved to: {detail_file}")
    
    # Make videos if requested
    if not args.no_video:
        videos_dir = pathlib.Path(args.path) / "edge_noise_videos"
        videos_dir.mkdir(exist_ok=True, parents=True)
        
        for ii, (rollout, Ta_is_unsafe) in enumerate(zip(rollouts[:args.max_videos], is_unsafes[:args.max_videos])):
            safe_rate, finish_rate, success_rate = rates[ii] * 100
            video_name = f"edge_noise_{args.edge_noise_std:.3f}_n{num_agents}_epi{ii:02d}_safe{safe_rate:.0f}_finish{finish_rate:.0f}_success{success_rate:.0f}"
            video_path = videos_dir / f"{stamp_str}_{video_name}.mp4"
            
            print(f"Generating video: {video_path.name}")
            env.render_video(rollout, video_path, Ta_is_unsafe, {}, dpi=args.dpi)


def main():
    parser = argparse.ArgumentParser(description="Test GCBF+ with edge feature noise in double_integrator environment")
    
    # Required arguments
    parser.add_argument("--path", type=str, required=True, help="Path to trained GCBF+ model")
    parser.add_argument("--area-size", type=float, required=True, help="Area size for the environment")
    
    # Environment parameters
    parser.add_argument("-n", "--num-agents", type=int, default=None, help="Number of agents (default: from config)")
    parser.add_argument("--obs", type=int, default=0, help="Number of obstacles")
    parser.add_argument("--max-step", type=int, default=None, help="Maximum steps per episode")
    parser.add_argument("--max-travel", type=float, default=None, help="Maximum travel distance")
    
    # Noise parameters
    parser.add_argument("--edge-noise-std", type=float, default=0.1, help="Standard deviation of Gaussian noise added to edge features")
    
    # Test parameters
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--epi", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--offset", type=int, default=0, help="Episode offset")
    parser.add_argument("--step", type=int, default=None, help="Model step to load (default: latest)")
    
    # Output parameters
    parser.add_argument("--log", action="store_true", help="Save results to log files")
    parser.add_argument("--no-video", action="store_true", help="Skip video generation")
    parser.add_argument("--max-videos", type=int, default=3, help="Maximum number of videos to generate")
    parser.add_argument("--dpi", type=int, default=100, help="DPI for video rendering")
    
    # System parameters
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (disable JIT)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.path):
        raise ValueError(f"Model path does not exist: {args.path}")
    
    if not os.path.exists(os.path.join(args.path, "config.yaml")):
        raise ValueError(f"Config file not found: {os.path.join(args.path, 'config.yaml')}")
    
    if args.edge_noise_std < 0:
        raise ValueError("Edge noise standard deviation must be non-negative")
    
    test_gcbf_plus_with_edge_noise(args)


if __name__ == "__main__":
    main()