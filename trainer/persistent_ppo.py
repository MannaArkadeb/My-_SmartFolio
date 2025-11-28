"""
Persistent PPO: Fine-tuning with parameter regularization toward baseline.

This module implements a PPO variant that adds an L2 penalty to keep fine-tuned
parameters close to a baseline checkpoint during training. This is inspired by
persistence mechanisms that prevent catastrophic forgetting during incremental learning.

Usage:
    model = PersistentPPO(
        policy='MlpPolicy',
        env=env,
        baseline_checkpoint='path/to/baseline.zip',
        persistency_lambda=1e-4,
        **standard_ppo_params
    )
    model.learn(total_timesteps=10000)
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.vec_env import VecEnv

SelfPersistentPPO = TypeVar("SelfPersistentPPO", bound="PersistentPPO")


class PersistentPPO(PPO):
    """
    PPO with L2 regularization toward baseline model parameters.

    Adds a persistency penalty to the training loss:
        total_loss = ppo_loss + lambda * ||theta - theta_baseline||_2^2

    This prevents catastrophic forgetting when fine-tuning on new data or tasks.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from
    :param baseline_checkpoint: Path to baseline PPO checkpoint (.zip file).
        If None, no persistency penalty is applied.
    :param persistency_lambda: Coefficient for the L2 penalty toward baseline.
        Higher values keep parameters closer to baseline. Default: 0.0 (disabled).
    :param persistency_device: Device to store baseline parameters ('cpu' or 'cuda').
        Default: 'cpu' to save GPU memory.
    :param kwargs: Additional PPO parameters
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        baseline_checkpoint: Optional[str] = None,
        persistency_lambda: float = 0.0,
        persistency_device: str = "cpu",
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        self.baseline_checkpoint = baseline_checkpoint
        self.persistency_lambda = persistency_lambda
        self.persistency_device = persistency_device
        self.baseline_params: Dict[str, th.Tensor] = {}
        self._persistency_enabled = False

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

        # Load baseline parameters after policy is initialized
        if self.baseline_checkpoint and self.persistency_lambda > 0:
            self._load_baseline_parameters()

    def _load_baseline_parameters(self) -> None:
        """Load baseline model parameters for persistency penalty."""
        if not self.baseline_checkpoint:
            return

        baseline_path = Path(self.baseline_checkpoint)
        if not baseline_path.exists():
            warnings.warn(
                f"Baseline checkpoint not found: {baseline_path}. "
                "Persistency penalty will be disabled.",
                UserWarning,
            )
            self.persistency_lambda = 0.0
            return

        try:
            # Load baseline model (without creating env)
            baseline_model = PPO.load(
                self.baseline_checkpoint,
                device=self.persistency_device,
                print_system_info=False,
            )

            # Extract and store baseline parameters
            baseline_state = baseline_model.policy.state_dict()
            for name, param in baseline_state.items():
                # Store as detached tensors on specified device
                self.baseline_params[name] = param.detach().clone().to(self.persistency_device)

            self._persistency_enabled = True
            if self.verbose > 0:
                print(
                    f"[PersistentPPO] Loaded baseline from {baseline_path} "
                    f"({len(self.baseline_params)} parameters)"
                )
                print(
                    f"[PersistentPPO] Persistency lambda: {self.persistency_lambda:.2e}"
                )

        except Exception as e:
            warnings.warn(
                f"Failed to load baseline checkpoint: {e}. "
                "Persistency penalty will be disabled.",
                UserWarning,
            )
            self.persistency_lambda = 0.0
            self.baseline_params.clear()

    def _compute_persistency_loss(self) -> th.Tensor:
        """
        Compute L2 penalty between current policy parameters and baseline.

        Returns:
            Scalar tensor with the persistency loss.
        """
        if not self._persistency_enabled or self.persistency_lambda == 0.0:
            return th.tensor(0.0, device=self.device)

        penalty = th.tensor(0.0, device=self.device)
        matched_params = 0

        current_state = self.policy.state_dict()
        for name, current_param in current_state.items():
            baseline_param = self.baseline_params.get(name)
            if baseline_param is None:
                continue

            # Move baseline to same device as current param for computation
            baseline_on_device = baseline_param.to(current_param.device)

            # Check shape compatibility
            if current_param.shape != baseline_on_device.shape:
                if self.verbose > 0:
                    warnings.warn(
                        f"Shape mismatch for parameter '{name}': "
                        f"current {current_param.shape} vs baseline {baseline_on_device.shape}. "
                        "Skipping this parameter.",
                        UserWarning,
                    )
                continue

            # Compute L2 penalty
            diff = current_param - baseline_on_device
            penalty = penalty + th.sum(diff ** 2)
            matched_params += 1

        if matched_params == 0 and self.verbose > 0:
            warnings.warn(
                "No matching parameters found between current policy and baseline. "
                "Persistency penalty is zero.",
                UserWarning,
            )

        return penalty

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.

        Overrides PPO.train() to add persistency penalty to the loss.
        """
        # Switch to train mode (affects dropout, batchnorm, etc.)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]

        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        persistency_losses = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )

                # Value loss using the TD(gae_lambda) target
                value_loss = th.nn.functional.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # ===== PERSISTENCY PENALTY =====
                persistency_loss = self._compute_persistency_loss()
                persistency_losses.append(persistency_loss.item())

                # Total loss with persistency penalty
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.persistency_lambda * persistency_loss
                )

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

                approx_kl_divs.append(
                    th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy()
                )

            self._n_updates += 1
            if not continue_training:
                break

            # Early stopping based on KL divergence
            if self.target_kl is not None:
                if np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                    if self.verbose > 0:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}"
                        )
                    continue_training = False

        # Log training metrics
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logging
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)

        # Log persistency metrics
        if self._persistency_enabled:
            self.logger.record("train/persistency_loss", np.mean(persistency_losses))
            self.logger.record("train/persistency_lambda", self.persistency_lambda)

        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def save(
        self,
        path: Union[str, Path],
        exclude: Optional[list[str]] = None,
        include: Optional[list[str]] = None,
    ) -> None:
        """
        Save the model. Baseline parameters are not saved (only metadata).

        :param path: path to save the model
        :param exclude: list of parameter names to exclude from save
        :param include: list of parameter names to include in save
        """
        # Don't save baseline params (they're loaded from checkpoint)
        super().save(path, exclude=exclude, include=include)

    def _excluded_save_params(self) -> list[str]:
        """Exclude baseline parameters from being saved."""
        excluded = super()._excluded_save_params()
        # Don't save the baseline parameters dict
        excluded.extend(["baseline_params"])
        return excluded

    @classmethod
    def load(
        cls: Type[SelfPersistentPPO],
        path: Union[str, Path],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs: Any,
    ) -> SelfPersistentPPO:
        """
        Load a model from file. Baseline checkpoint must be specified in kwargs.

        :param path: path to the saved model
        :param env: Environment to run the model on
        :param device: Device on which the model should be loaded
        :param custom_objects: Dictionary of custom objects to replace
        :param print_system_info: Whether to print system info
        :param force_reset: Force a call to reset() before training
        :param kwargs: Additional arguments (must include baseline_checkpoint and
            persistency_lambda if persistency is desired)
        :return: Loaded model instance
        """
        # Load the base model
        model = super().load(
            path,
            env=env,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
            force_reset=force_reset,
            **kwargs,
        )

        # Restore persistency settings if provided
        if "baseline_checkpoint" in kwargs:
            model.baseline_checkpoint = kwargs["baseline_checkpoint"]
        if "persistency_lambda" in kwargs:
            model.persistency_lambda = kwargs["persistency_lambda"]
        if "persistency_device" in kwargs:
            model.persistency_device = kwargs["persistency_device"]

        # Reload baseline parameters
        if model.baseline_checkpoint and model.persistency_lambda > 0:
            model._load_baseline_parameters()

        return model
