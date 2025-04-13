# Design department agent class for managing design processes

import numpy as np
from typing import List, Dict, Tuple

class DesignDepartment:
    """Design department agent class"""
    
    def __init__(self, shared_factory, knowledge_transfer_rate):
        self.completed_designs = 0  # Number of completed designs
        self.ongoing_designs = {}   # Current designs in progress {order_id: progress}
        self.design_queue = []      # Design queue
        self.design_costs = {}      # Design costs per order
        self.knowledge_transfer_rate = knowledge_transfer_rate
        self.shared_factory = shared_factory
        self.max_concurrent_designs = 0  # Set in factory_env to match production lines
        
    def set_max_concurrent_designs(self, num_production_lines: int):
        """Set maximum number of concurrent designs"""
        self.max_concurrent_designs = num_production_lines
        
    def assign_order_to_line(self, 
                            order,
                            production_lines: List,
                            current_time: int) -> Tuple[int, float]:
        # Calculate expected design completion time
        expected_design_duration = order.duration * order.era_ratio  
        expected_completion_time = current_time + expected_design_duration
        
        best_line = None
        min_cost = float('inf')
        
        # Check for available lines at expected completion time
        available_lines = []
        earliest_completion = float('inf')
        
        for line in production_lines:
            if line.is_working:
                continue
                
            if line.is_available_at(expected_completion_time):
                available_lines.append(line)
            else:
                completion_time = line.get_next_available_time()
                if completion_time < earliest_completion:
                    earliest_completion = completion_time
        
        # If no lines available, select from earliest completion time
        candidate_lines = available_lines if available_lines else \
                         [line for line in production_lines 
                          if not line.is_working and  
                          line.get_next_available_time() == earliest_completion]
        
        # Select line with minimum u_j × r_it from candidates
        for line in candidate_lines:
            mismatch = (order.quality_req - line.quality_index) ** 2 / 100
            cost = mismatch * line.completed_orders
            
            if cost < min_cost:
                min_cost = cost
                best_line = line
        
        return best_line.id if best_line else None, min_cost
        
    def calculate_design_duration(self, order, knowledge_level: float) -> int:
        """Calculate design duration with knowledge effect"""
        base_duration = order.duration * order.era_ratio
        beta = np.random.uniform(-0.1, 0.1)
        duration = base_duration * (1 + beta) * knowledge_level  
        return max(1, int(duration))

    def start_design(self, order, current_time: int, knowledge_level: float):
        """Start design process for an order"""
        duration = self.calculate_design_duration(order, knowledge_level)
        
        if len(self.ongoing_designs) >= self.max_concurrent_designs:
            self.design_queue.append({
                'order': order,
                'knowledge_level': knowledge_level,
                'arrival_time': current_time
            })
            print(f"Order {order.id} added to design queue (queue length: {len(self.design_queue)})")
            return
            
        print(f"Starting design for order {order.id}, duration: {duration} "
              f"(ongoing designs: {len(self.ongoing_designs)})")
        self.ongoing_designs[order.id] = {
            'order': order,
            'progress': 0,
            'duration': duration,
            'start_time': current_time
        }
        order.design_start_time = current_time
        
    def step(self, current_time: int) -> List[int]:
        """Time step update"""
        completed_orders = []
        
        # Ensure not exceeding max concurrent designs
        assert len(self.ongoing_designs) <= self.max_concurrent_designs, \
               f"Too many ongoing designs: {len(self.ongoing_designs)}"
        
        # Update all ongoing designs
        for order_id, design_info in list(self.ongoing_designs.items()):
            design_info['progress'] += 1
            
            # Check for completion
            if design_info['progress'] >= design_info['duration']:
                completed_orders.append(order_id)
                self.completed_designs += 1
                del self.ongoing_designs[order_id]
                
                # Notify SharedFactory of design completion
                self.shared_factory.completed_designs += 1
                
                # Start new design from queue
                if self.design_queue:
                    next_design = self.design_queue.pop(0)
                    self.start_design(
                        next_design['order'], 
                        current_time,
                        next_design['knowledge_level']
                    )
                order = design_info['order']
                order.design_completion_time = current_time
        
        return completed_orders
        
    def get_queue_length(self) -> int:
        """Get design queue length"""
        return len(self.design_queue)
        
    def get_total_orders(self) -> int:
        """Get total orders in design department (ongoing + queued)"""
        return len(self.ongoing_designs) + len(self.design_queue)
    
