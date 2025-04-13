# src/agents/production_line.py

import numpy as np
from typing import List, Dict, Optional

class ProductionLine:
    """Production line agent class"""
    
    def __init__(self, 
                 line_id: int,
                 quality_index: int,
                 base_inconvenience: float,
                 inconvenience_level: float):  
        self.id = line_id
        self.quality_index = quality_index  # Production line quality index
        self.base_inconvenience = base_inconvenience
        self.inconvenience_level = inconvenience_level
        
        # State variables
        self.is_working = False
        self.current_order = None
        self.received_orders = 0  # r_it defined as received_orders
        self.completed_orders = 0
        self.production_schedule = {}  # {time: order_id}
        self.inconvenience_costs = []
        
    def calculate_production_duration(self, order) -> int:
        """Calculate production duration"""
        base_duration = order.duration * (1 - order.era_ratio)
        disturbance = np.random.uniform(-0.1, 0.1)
        return max(1, int(base_duration * (1 + disturbance)))  # Ensure at least 1 time unit
        
    def calculate_inconvenience_cost(self) -> float:
        """Calculate inconvenience cost using received_orders"""
        prev_received = self.received_orders - 1
        return (self.received_orders ** self.inconvenience_level * 
                self.base_inconvenience * 
                (self.received_orders - prev_received))
        
    def start_production(self, order, current_time: int):
        """Start production for an order"""
        if self.is_working:
            raise ValueError(f"Production line {self.id} is already working on order {self.current_order.id}")
        
        if order is None:
            raise ValueError(f"Cannot start production with None order")
        
        self.is_working = True
        self.current_order = order
        self.received_orders += 1
        
        # Calculate production time
        production_duration = self.calculate_production_duration(order)
        print(f"Line {self.id} starting production for order {order.id}, duration: {production_duration}")
        
        # Update production schedule
        end_time = current_time + production_duration
        self.production_schedule[end_time] = order.id
        
        # Calculate and record inconvenience cost
        inconvenience_cost = self.calculate_inconvenience_cost()
        self.inconvenience_costs.append(inconvenience_cost)
        
        order.production_start_time = current_time
        
    def is_available_at(self, time: int) -> bool:
        """Check if line is available at specified time"""
        return not any(t >= time for t in self.production_schedule.keys())

    def get_next_available_time(self) -> int:
        """Get next available time"""
        if not self.production_schedule:
            return 0
        return max(self.production_schedule.keys())
        
    def step(self, current_time: int) -> Optional[int]:
        """Time step update"""
        # Check for completed orders
        if current_time in self.production_schedule:
            completed_order_id = self.production_schedule[current_time]
            
            if self.current_order:  
                self.current_order.completion_time = current_time
            
            # Update status
            self.is_working = False
            self.completed_orders += 1
            
            # Clear order
            self.current_order = None
            del self.production_schedule[current_time]
            
            return completed_order_id
        
        return None

    def get_current_inconvenience_cost(self) -> float:
        """Get current inconvenience cost"""
        if self.inconvenience_costs:
            return self.inconvenience_costs[-1]
        return 0.0