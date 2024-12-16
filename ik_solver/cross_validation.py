import torch
import numpy as np
from .mappo import MAPPOAgent
import logging
class CrossValidator:
    def __init__(self, env, config, k_folds=5):
        """
        Initialize cross-validator for MAPPO.
        
        Args:
            env: Training environment
            config: Configuration dictionary
            k_folds: Number of cross-validation folds
        """
        self.env = env
        self.config = config
        self.k_folds = k_folds
        self.current_fold = 0
        self.validation_results = []
        
        # Initialize separate validation environments
        self.validation_envs = [
            self._create_validation_env() for _ in range(k_folds)
        ]
        
        self.logger = logging.getLogger(__name__)

    def _create_validation_env(self):
        """Create a separate environment for validation"""
        return self.env.__class__(**self.env.get_params())

    def validate_model(self, agent, fold_idx):
        """
        Validate agent on a specific fold.
        
        Args:
            agent: MAPPO agent to validate
            fold_idx: Index of the validation fold
        """
        validation_env = self.validation_envs[fold_idx]
        num_validation_episodes = self.config.get('validation_episodes', 10)
        
        validation_metrics = {
            'rewards': [],
            'success_rate': [],
            'joint_errors': [],
            'episode_lengths': []
        }
        
        for episode in range(num_validation_episodes):
            state = validation_env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            episode_errors = []
            
            while not done and episode_steps < self.config.get('max_steps_per_episode', 1000):
                # Get actions without exploration noise
                with torch.no_grad():
                    actions, _ = agent.get_actions(state)
                
                next_state, rewards, done, info = validation_env.step(actions)
                
                episode_reward += sum(rewards)
                episode_steps += 1
                if 'joint_errors' in info:
                    episode_errors.append(info['joint_errors'])
                
                state = next_state
            
            validation_metrics['rewards'].append(episode_reward)
            validation_metrics['success_rate'].append(info.get('success', False))
            validation_metrics['episode_lengths'].append(episode_steps)
            if episode_errors:
                validation_metrics['joint_errors'].append(np.mean(episode_errors))
        
        return {
            'mean_reward': np.mean(validation_metrics['rewards']),
            'std_reward': np.std(validation_metrics['rewards']),
            'success_rate': np.mean(validation_metrics['success_rate']),
            'mean_episode_length': np.mean(validation_metrics['episode_lengths']),
            'mean_joint_error': np.mean(validation_metrics['joint_errors']) if validation_metrics['joint_errors'] else None
        }

    def train_and_validate(self):
        """Perform k-fold cross-validation"""
        fold_results = []
        
        for fold in range(self.k_folds):
            self.logger.info(f"Starting fold {fold + 1}/{self.k_folds}")
            
            # Initialize new agent for this fold
            agent = MAPPOAgent(self.env, self.config)
            
            # Train agent
            metrics = agent.train()
            
            # Validate on this fold
            val_metrics = self.validate_model(agent, fold)
            
            fold_results.append({
                'fold': fold,
                'training_metrics': metrics,
                'validation_metrics': val_metrics
            })
            
            self.logger.info(f"Fold {fold + 1} Results:")
            self.logger.info(f"Training Success Rate: {metrics.get('success_rate', {}).get('overall', 0):.4f}")
            self.logger.info(f"Validation Success Rate: {val_metrics['success_rate']:.4f}")
            self.logger.info(f"Validation Mean Reward: {val_metrics['mean_reward']:.4f}")
        
        return self.analyze_results(fold_results)

    def analyze_results(self, fold_results):
        """Analyze cross-validation results"""
        analysis = {
            'training': {
                'success_rates': [],
                'mean_rewards': [],
                'joint_errors': []
            },
            'validation': {
                'success_rates': [],
                'mean_rewards': [],
                'joint_errors': []
            }
        }
        
        for result in fold_results:
            # Training metrics
            train_metrics = result['training_metrics']
            analysis['training']['success_rates'].append(
                train_metrics.get('success_rate', {}).get('overall', 0)
            )
            analysis['training']['mean_rewards'].append(
                train_metrics.get('rewards', {}).get('overall_mean', 0)
            )
            if 'joint_errors' in train_metrics:
                analysis['training']['joint_errors'].append(
                    train_metrics['joint_errors'].get('overall_average', 0)
                )
            
            # Validation metrics
            val_metrics = result['validation_metrics']
            analysis['validation']['success_rates'].append(val_metrics['success_rate'])
            analysis['validation']['mean_rewards'].append(val_metrics['mean_reward'])
            if val_metrics['mean_joint_error'] is not None:
                analysis['validation']['joint_errors'].append(val_metrics['mean_joint_error'])
        
        # Compute statistics
        results = {
            'training': {
                'mean_success_rate': np.mean(analysis['training']['success_rates']),
                'std_success_rate': np.std(analysis['training']['success_rates']),
                'mean_reward': np.mean(analysis['training']['mean_rewards']),
                'std_reward': np.std(analysis['training']['mean_rewards']),
                'mean_joint_error': np.mean(analysis['training']['joint_errors']),
                'std_joint_error': np.std(analysis['training']['joint_errors'])
            },
            'validation': {
                'mean_success_rate': np.mean(analysis['validation']['success_rates']),
                'std_success_rate': np.std(analysis['validation']['success_rates']),
                'mean_reward': np.mean(analysis['validation']['mean_rewards']),
                'std_reward': np.std(analysis['validation']['mean_rewards']),
                'mean_joint_error': np.mean(analysis['validation']['joint_errors']),
                'std_joint_error': np.std(analysis['validation']['joint_errors'])
            }
        }
        
        # Log results
        self.logger.info("\nCross-Validation Results:")
        self.logger.info(f"Training Success Rate: {results['training']['mean_success_rate']:.4f} ± {results['training']['std_success_rate']:.4f}")
        self.logger.info(f"Validation Success Rate: {results['validation']['mean_success_rate']:.4f} ± {results['validation']['std_success_rate']:.4f}")
        self.logger.info(f"Training Mean Reward: {results['training']['mean_reward']:.4f} ± {results['training']['std_reward']:.4f}")
        self.logger.info(f"Validation Mean Reward: {results['validation']['mean_reward']:.4f} ± {results['validation']['std_reward']:.4f}")
        
        return results

