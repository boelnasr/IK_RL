import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor
import queue
import os

def setup_distributed(rank, world_size):
    """Initialize distributed training for DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # Set the current process to use a specific GPU


class ParallelEnvWrapper:
    """Wrapper for running multiple environments in parallel with DDP support"""
    def __init__(self, env_fn, num_envs, device, rank=0, world_size=1):
        self.envs = [env_fn() for _ in range(num_envs)]
        self.num_envs = num_envs
        self.device = device
        self.executor = ThreadPoolExecutor(max_workers=num_envs)
        self.rank = rank
        self.world_size = world_size
        
        if world_size > 1:
            setup_distributed(rank, world_size)
        
    def reset(self):
        """Reset all environments in parallel"""
        futures = [self.executor.submit(env.reset) for env in self.envs]
        states = [f.result() for f in futures]
        
        # Synchronize across processes if using DDP
        if self.world_size > 1:
            states = self._sync_states(states)
            
        return self._process_states(states)
        
    def step(self, actions):
        """Step all environments in parallel"""
        futures = [
            self.executor.submit(env.step, action) 
            for env, action in zip(self.envs, actions)
        ]
        results = [f.result() for f in futures]
        
        next_states, rewards, dones, infos = zip(*results)
        
        # Synchronize across processes if using DDP
        if self.world_size > 1:
            next_states = self._sync_states(next_states)
            rewards = self._sync_tensor(self._to_tensor(rewards))
            dones = self._sync_tensor(self._to_tensor(dones))
        
        return (
            self._process_states(next_states),
            self._to_tensor(rewards),
            self._to_tensor(dones),
            infos
        )
    
    def _process_states(self, states):
        """Process states from all environments"""
        processed_states = []
        for state_list in zip(*states):
            processed_state = torch.stack([
                torch.from_numpy(state).float() for state in state_list
            ]).to(self.device)
            processed_states.append(processed_state)
        return processed_states
    
    def _to_tensor(self, array):
        """Convert array to tensor"""
        return torch.tensor(array, dtype=torch.float32).to(self.device)
    
    def _sync_states(self, states):
        """Synchronize states across processes"""
        if self.world_size > 1:
            gathered_states = [None] * self.world_size
            dist.all_gather_object(gathered_states, states)
            states = [s for gathered in gathered_states for s in gathered]
        return states
    
    def _sync_tensor(self, tensor):
        """Synchronize tensor across processes"""
        if self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= self.world_size
        return tensor
    
    def close(self):
        """Close all environments and cleanup"""
        for env in self.envs:
            env.close()
        self.executor.shutdown()
        if self.world_size > 1:
            dist.destroy_process_group()

class BatchProcessor:
    """Enhanced batch processing with multi-GPU support"""
    def __init__(self, device, num_agents, world_size=1, rank=0):
        self.device = device
        self.num_agents = num_agents
        self.batch_queue = mp.Queue()
        self.world_size = world_size
        self.rank = rank
        
        # Start worker processes for batch processing
        if world_size > 1:
            self.workers = []
            for i in range(world_size):
                p = mp.Process(target=self._batch_worker, args=(i,))
                p.start()
                self.workers.append(p)
        
    def process_batch(self, batch_states, batch_actions, batch_log_probs_old, 
                     batch_advantages, batch_returns):
        """Process batch data with multi-GPU support"""
        # Shard data across GPUs if using multiple
        if self.world_size > 1:
            shard_size = len(batch_states) // self.world_size
            start_idx = self.rank * shard_size
            end_idx = start_idx + shard_size if self.rank < self.world_size - 1 else len(batch_states)
            
            batch_states = batch_states[start_idx:end_idx]
            batch_actions = batch_actions[start_idx:end_idx]
            batch_log_probs_old = batch_log_probs_old[start_idx:end_idx]
            batch_advantages = batch_advantages[start_idx:end_idx]
            batch_returns = batch_returns[start_idx:end_idx]
        
        # Process in chunks for memory efficiency
        chunk_size = 1024  # Adjust based on GPU memory
        num_chunks = (len(batch_states) + chunk_size - 1) // chunk_size
        
        processed_data = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(batch_states))
            
            # Process chunk
            chunk_data = {
                'states': batch_states[start_idx:end_idx].to(self.device),
                'actions': batch_actions[start_idx:end_idx].to(self.device),
                'log_probs_old': batch_log_probs_old[start_idx:end_idx].to(self.device),
                'advantages': batch_advantages[start_idx:end_idx].to(self.device),
                'returns': batch_returns[start_idx:end_idx].to(self.device)
            }
            
            # Synchronize across GPUs if needed
            if self.world_size > 1:
                chunk_data = self._sync_chunk(chunk_data)
            
            processed_data.append(chunk_data)
        
        return processed_data
    
    def _sync_chunk(self, chunk_data):
        """Synchronize chunk data across GPUs"""
        if self.world_size > 1:
            for k, v in chunk_data.items():
                dist.all_reduce(v, op=dist.ReduceOp.SUM)
                chunk_data[k] = v / self.world_size
        return chunk_data
    
    def _batch_worker(self, rank):
        """Worker process for batch processing"""
        while True:
            try:
                batch = self.batch_queue.get()
                if batch is None:  # Poison pill
                    break
                processed_batch = self.process_batch(**batch)
                self.batch_queue.put(processed_batch)
            except Exception as e:
                print(f"Worker {rank} error: {e}")
                continue
    
    def cleanup(self):
        """Cleanup worker processes"""
        if self.world_size > 1:
            for _ in self.workers:
                self.batch_queue.put(None)
            for w in self.workers:
                w.join()

                