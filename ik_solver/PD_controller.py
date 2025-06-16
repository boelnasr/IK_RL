import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class PDController:
    def __init__(self, kp=1.0, kd=0.1, dt=0.01, stability_threshold=0.05):
        """
        Initialize PD controller with tracking and visualization capabilities.
        
        Args:
            kp (float): Proportional gain
            kd (float): Derivative gain
            dt (float): Time step for derivative calculation
            stability_threshold (float): Threshold for error smoothness
        """
        self.kp = kp
        self.kd = kd
        self.dt = dt
        self.previous_error = None
        self.stability_threshold = stability_threshold
        
        # Performance tracking
        self.corrections_history = []
        self.errors_history = []
        self.weight_history = []  # Add proper weight history tracking
        self.weight = 0.3  # PD influence weight
        
        # Stability metrics
        self.stability_metrics = {
            'error_smoothness': 0.0,
            'control_smoothness': 0.0,
            'overall_stability': 0.0
        }
        
        # For adaptive weight adjustment
        self.performance_window = deque(maxlen=100)  # Window for performance tracking
        
    def compute(self, error):
        """
        Compute PD control signal with tracking.
        
        Args:
            error (float): Current error value
            
        Returns:
            float: Control signal
        """
        if self.previous_error is None:
            self.previous_error = error
            derivative = 0
        else:
            derivative = (error - self.previous_error) / self.dt
        
        control = self.kp * error + self.kd * derivative
        
        # Track history
        self.corrections_history.append(control)
        self.errors_history.append(error)
        
        # Update previous error
        self.previous_error = error
        
        return control
    
    def reset(self):
        """Reset the controller state."""
        self.previous_error = None
    
    def track_performance(self):
        """
        Calculate performance metrics based on recent history.
        
        Returns:
            dict: Dictionary of stability metrics
        """
        if len(self.errors_history) < 2 or len(self.corrections_history) < 2:
            return self.stability_metrics
        
        # Use recent history for calculations (last 100 steps)
        recent_errors = self.errors_history[-100:]
        recent_corrections = self.corrections_history[-100:]
        
        if len(recent_errors) > 1:
            # Calculate error smoothness (lower is better)
            error_diffs = np.abs(np.diff(recent_errors))
            self.stability_metrics['error_smoothness'] = np.nanmean(error_diffs)
        
        if len(recent_corrections) > 1:
            # Calculate control smoothness (lower is better)
            control_diffs = np.abs(np.diff(recent_corrections))
            self.stability_metrics['control_smoothness'] = np.nanmean(control_diffs)
        
        # Calculate overall stability (higher is better)
        if recent_errors:
            self.stability_metrics['overall_stability'] = np.exp(-np.nanmean(np.abs(recent_errors)))
        
        return self.stability_metrics
    
    def adapt_weight(self):
        """
        Adapt the PD influence weight based on performance metrics.
        
        Returns:
            float: New PD weight
        """
        # Update performance metrics
        self.track_performance()
        
        # Adapt weight based on error smoothness
        if self.stability_metrics['error_smoothness'] > self.stability_threshold:
            # Increase PD influence when errors are not smooth enough
            self.weight = min(self.weight * 1.05, 0.8)  # Upper limit of 0.8
        else:
            # Decrease PD influence when errors are smooth enough
            self.weight = max(self.weight * 0.98, 0.1)  # Lower limit of 0.1
        
        # Track weight history - THIS IS THE KEY ADDITION
        self.weight_history.append(self.weight)
        
        return self.weight
    
    def visualize_performance(self, save_path=None, agent_idx=None):
        """
        Visualize controller performance over time.
        
        Args:
            save_path (str, optional): Path to save the visualization
            agent_idx (int, optional): Agent index for labeling
            
        Returns:
            matplotlib.figure.Figure: Figure object if not saving to file
        """
        try:
            fig, axes = plt.subplots(4, 1, figsize=(12, 16), dpi=300)
            
            title_suffix = f" (Agent {agent_idx})" if agent_idx is not None else ""
            
            # Plot error history
            if self.errors_history:
                steps = range(len(self.errors_history))
                axes[0].plot(steps, self.errors_history, label='Error', color='red', alpha=0.7)
                axes[0].set_title(f'Error Over Time{title_suffix}')
                axes[0].set_xlabel('Step')
                axes[0].set_ylabel('Error Value')
                axes[0].grid(True, alpha=0.3)
                axes[0].legend()
            
            # Plot correction history
            if self.corrections_history:
                steps = range(len(self.corrections_history))
                axes[1].plot(steps, self.corrections_history, label='PD Correction', color='blue', alpha=0.7)
                axes[1].set_title(f'PD Correction Over Time{title_suffix}')
                axes[1].set_xlabel('Step')
                axes[1].set_ylabel('Correction Value')
                axes[1].grid(True, alpha=0.3)
                axes[1].legend()
            
            # Plot weight adaptation over time - NOW USING ACTUAL WEIGHT HISTORY
            if self.weight_history:
                steps = range(len(self.weight_history))
                axes[2].plot(steps, self.weight_history, label='PD Weight', color='green', linewidth=2)
                axes[2].set_title(f'PD Weight Adaptation Over Episode{title_suffix}')
                axes[2].set_xlabel('Step')
                axes[2].set_ylabel('Weight Value')
                axes[2].set_ylim(0, 1)  # Weight is between 0 and 1
                axes[2].grid(True, alpha=0.3)
                axes[2].legend()
                
                # Add text showing weight change statistics
                if len(self.weight_history) > 1:
                    initial_weight = self.weight_history[0]
                    final_weight = self.weight_history[-1]
                    weight_change = final_weight - initial_weight
                    mean_weight = np.mean(self.weight_history)
                    std_weight = np.std(self.weight_history)
                    
                    stats_text = (f'Initial: {initial_weight:.3f}\n'
                                f'Final: {final_weight:.3f}\n'
                                f'Change: {weight_change:+.3f}\n'
                                f'Mean: {mean_weight:.3f}\n'
                                f'Std: {std_weight:.3f}')
                    
                    axes[2].text(0.02, 0.95, stats_text,
                               transform=axes[2].transAxes,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Plot phase portrait (error vs. correction)
            if len(self.errors_history) > 1 and len(self.corrections_history) > 1:
                min_len = min(len(self.errors_history), len(self.corrections_history))
                errors = self.errors_history[-min_len:]
                corrections = self.corrections_history[-min_len:]
                
                # Create color gradient to show time progression
                colors = plt.cm.viridis(np.linspace(0, 1, min_len))
                
                # Plot with color gradient
                for i in range(min_len - 1):
                    axes[3].plot(errors[i:i+2], corrections[i:i+2], 
                               color=colors[i], alpha=0.7, linewidth=2)
                
                # Highlight start and end points
                axes[3].scatter(errors[0], corrections[0], 
                              color='green', s=100, marker='o', 
                              label='Start', edgecolors='black', linewidth=2, zorder=5)
                axes[3].scatter(errors[-1], corrections[-1], 
                              color='red', s=100, marker='s', 
                              label='End', edgecolors='black', linewidth=2, zorder=5)
                
                axes[3].set_title(f'Phase Portrait (Error vs. Correction){title_suffix}')
                axes[3].set_xlabel('Error')
                axes[3].set_ylabel('Correction')
                axes[3].grid(True, alpha=0.3)
                axes[3].legend()
            
            plt.tight_layout()
            
            # Save or return the figure
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)
                return None
            else:
                return fig
                
        except ImportError:
            print("Matplotlib is required for visualization")
            return None
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            return None
    
    def get_metrics(self):
        """
        Get current controller metrics.
        
        Returns:
            dict: Dictionary of controller metrics
        """
        def safe_nanmean(arr):
            if len(arr) == 0:
                return 0.0
            result = np.nanmean(arr)
            return result if not np.isnan(result) else 0.0
        
        def safe_nanstd(arr):
            if len(arr) == 0:
                return 0.0
            result = np.nanstd(arr)
            return result if not np.isnan(result) else 0.0
        
        metrics = {
            'kp': self.kp,
            'kd': self.kd,
            'weight': self.weight,
            'weight_history_length': len(self.weight_history),
            'weight_mean': safe_nanmean(self.weight_history) if self.weight_history else self.weight,
            'weight_std': safe_nanstd(self.weight_history) if self.weight_history else 0.0,
            'stability': self.stability_metrics.copy(),
            'recent_error_mean': safe_nanmean(self.errors_history[-20:]),
            'recent_error_std': safe_nanstd(self.errors_history[-20:]),
            'recent_correction_mean': safe_nanmean(self.corrections_history[-20:]),
            'corrections_count': len(self.corrections_history),
            'total_error': np.sum(np.abs(self.errors_history)) if self.errors_history else 0.0
        }
        return metrics
    def clear_history(self):
        """Clear the tracking history (useful for new episodes)."""
        self.corrections_history.clear()
        self.errors_history.clear()
        self.weight_history.clear()  # Also clear weight history
        self.stability_metrics = {
            'error_smoothness': 0.0,
            'control_smoothness': 0.0,
            'overall_stability': 0.0
        }
    
    def get_recent_performance(self, window_size=50):
        """
        Get performance metrics for the most recent window.
        
        Args:
            window_size (int): Size of the recent window
            
        Returns:
            dict: Recent performance metrics
        """
        if len(self.errors_history) < window_size:
            return self.get_metrics()
        
        recent_errors = self.errors_history[-window_size:]
        recent_corrections = self.corrections_history[-window_size:]
        recent_weights = self.weight_history[-window_size:] if len(self.weight_history) >= window_size else self.weight_history
        
        recent_metrics = {
            'mean_error': np.mean(recent_errors),
            'std_error': np.std(recent_errors),
            'max_error': np.max(np.abs(recent_errors)),
            'mean_correction': np.mean(recent_corrections),
            'std_correction': np.std(recent_corrections),
            'max_correction': np.max(np.abs(recent_corrections)),
            'mean_weight': np.mean(recent_weights) if recent_weights else self.weight,
            'weight_trend': np.polyfit(range(len(recent_weights)), recent_weights, 1)[0] if len(recent_weights) > 1 else 0,
            'error_trend': np.polyfit(range(window_size), recent_errors, 1)[0],  # Slope of error trend
            'correction_trend': np.polyfit(range(window_size), recent_corrections, 1)[0]  # Slope of correction trend
        }
        
        return recent_metrics