"""Tests for RLlib training configuration."""

import pytest
import numpy as np

# NumPy 2.0 compatibility - restore deprecated aliases used by RLlib
_NP_COMPAT_ALIASES = {
    'float_': np.float64,
    'int_': np.int64,
    'bool_': np.bool_,
    'object_': np.object_,
}
for _alias, _dtype in _NP_COMPAT_ALIASES.items():
    if not hasattr(np, _alias):
        setattr(np, _alias, _dtype)


class TestRLlibConfig:
    """Test RLlib configuration builds correctly."""
    
    def test_ppo_config_builds(self):
        """Test that PPO config builds without errors."""
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.models import ModelCatalog
        from ray.tune.registry import register_env
        from aegis.core.rules import GameConfig
        from aegis.env.pettingzoo_env import AegisEnv
        from aegis.train.rllib_ppo import ActionMaskModel, AegisRLlibEnv
        
        # Initialize Ray
        ray.init(ignore_reinit_error=True, local_mode=True)
        
        try:
            # Register custom model
            ModelCatalog.register_custom_model("action_mask_model", ActionMaskModel)
            
            # Create environment
            env_config = {
                "num_survivors": 2,
                "num_impostors": 1,
                "max_ticks": 50,
            }
            
            def env_creator(cfg):
                return AegisRLlibEnv({
                    "env_config": env_config,
                    "rewards_config": {},
                })
            
            register_env("aegis_test", env_creator)
            
            # Get sample spaces from the RLlib env wrapper
            sample_env = env_creator({})
            obs_space = sample_env.observation_space
            act_space = sample_env.action_space
            
            # Define policies
            policies = {
                "survivor_policy": (None, obs_space, act_space, {}),
                "impostor_policy": (None, obs_space, act_space, {}),
            }
            
            # Build config
            ppo_config = (
                PPOConfig()
                .api_stack(
                    enable_rl_module_and_learner=False,
                    enable_env_runner_and_connector_v2=False,
                )
                .environment(
                    env="aegis_test",
                    env_config={},
                )
                .framework("torch")
                .env_runners(
                    num_env_runners=0,  # Local mode
                    num_envs_per_env_runner=1,
                )
                .training(
                    train_batch_size=200,
                    minibatch_size=50,
                    num_sgd_iter=2,
                    lr=0.0003,
                    model={
                        "custom_model": "action_mask_model",
                        "fcnet_hiddens": [64, 64],
                        "fcnet_activation": "relu",
                        "_disable_preprocessor_api": True,
                    },
                )
                .multi_agent(
                    policies=policies,
                    policy_mapping_fn=lambda agent_id, *args, **kwargs: (
                        "impostor_policy" if int(agent_id.split("_")[1]) < 1
                        else "survivor_policy"
                    ),
                )
                .resources(num_gpus=0)
            )
            
            # This should not raise an error
            algo = ppo_config.build_algo()
            
            # Run one quick training step
            result = algo.train()
            
            assert result is not None
            assert "env_runners" in result or "sampler_results" in result or "info" in result
            
            algo.stop()
            
        finally:
            ray.shutdown()
    
    def test_action_mask_model_forward(self):
        """Test that the custom action mask model can forward pass."""
        import torch
        import numpy as np
        from gymnasium import spaces
        
        from aegis.train.rllib_ppo import ActionMaskModel
        from aegis.core.rules import GameConfig
        from aegis.env.pettingzoo_env import AegisEnv
        
        # Create env and get spaces
        config = GameConfig(num_survivors=2, num_impostors=1, max_ticks=50)
        env = AegisEnv(config=config)
        obs_space = env.observation_space("agent_0")
        act_space = env.action_space("agent_0")
        
        # Create model
        model = ActionMaskModel(
            obs_space=obs_space,
            action_space=act_space,
            num_outputs=act_space.n,
            model_config={"fcnet_hiddens": [64, 64], "fcnet_activation": "relu"},
            name="test_model",
        )
        
        # Create a sample observation
        obs, _ = env.reset(seed=42)
        sample_obs = obs["agent_0"]
        
        # Convert observation to tensors with batch dimension
        obs_tensors = {}
        for key, value in sample_obs.items():
            if isinstance(value, np.ndarray):
                obs_tensors[key] = torch.tensor(value, dtype=torch.float32).unsqueeze(0)
            else:
                obs_tensors[key] = torch.tensor([value], dtype=torch.float32).unsqueeze(0)
        
        # Forward pass
        input_dict = {"obs": obs_tensors}
        logits, state = model.forward(input_dict, [], None)
        
        assert logits is not None
        assert logits.shape == (1, act_space.n)
        
        # Check value function
        value = model.value_function()
        assert value is not None
        assert value.shape == (1,)
    
    def test_env_config_loads(self):
        """Test that environment config loads correctly from YAML."""
        from aegis.train.rllib_ppo import load_config, create_env
        
        config = load_config("aegis/train/configs/base.yaml")
        
        assert "env" in config
        assert "rewards" in config
        assert "training" in config
        
        env_config = config.get("env", {})
        rewards_config = config.get("rewards", {})
        
        # Test creating environment
        env = create_env(env_config, rewards_config)
        
        assert env is not None
        assert env.config.reward_proximity_penalty == rewards_config.get("proximity_penalty", -0.01)
        
        # Test reset and step
        obs, info = env.reset(seed=42)
        assert len(obs) == env.config.num_agents
        
        # Random actions
        actions = {name: 0 for name in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        assert len(obs) == env.config.num_agents


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

