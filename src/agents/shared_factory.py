import numpy as np

class SharedFactory:
    """Shared factory agent class for managing global factory state"""
    
    def __init__(self, 
                 alpha1: float = 0.5,  # Cost-profit ratio weight
                 alpha2: float = 0.3,  # Equipment utilization weight
                 alpha3: float = 0.2,  # Delay rate weight
                 knowledge_transfer_rate: float = 0.02):  # Knowledge transfer rate
        
        # Weight parameters
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.knowledge_transfer_rate = knowledge_transfer_rate
        
        # State variables
        self.accepted_orders = []
        self.delayed_orders = 0
        self.total_cost = 0.0
        self.total_profit = 0.0
        self.knowledge_effect = 1.0  
        self.equipment_efficiency = 0.0
        self.completed_designs = 0  # For knowledge effect calculation
        self.completed_orders = 0   # For delay rate calculation
        self.total_delay_time = 0  
        self.acceptance_threshold = 0.0
        self.min_orders = 0
        
    def update_knowledge_effect(self):
        """Update knowledge effect"""
        if self.completed_designs == 0:
            self.knowledge_effect = 1.0
        else:
            # Direct calculation of knowledge effect
            self.knowledge_effect = self.completed_designs ** (-self.knowledge_transfer_rate)
    
    def _calculate_utility(self, order) -> float:
        """Calculate utility function"""
        if not self.accepted_orders:
            return 1.0
            
        cp_ratio = max(0.1, self.total_profit / (self.total_cost + 1e-6))
        ee = max(0.1, self.equipment_efficiency)
        delay_ratio = self.delayed_orders / (len(self.accepted_orders) + 1e-6)
        
        utility = (cp_ratio ** self.alpha1 * 
                  ee ** self.alpha2 * 
                  (1 / (1 + delay_ratio)) ** self.alpha3)
                  
        return utility
        
    def decide_order_acceptance(self, order) -> bool:
        """Accept all orders"""
        self.accepted_orders.append(order)
        return True
        
    def update_equipment_efficiency(self, working_lines: int, total_lines: int):
        """Update equipment efficiency"""
        self.equipment_efficiency = working_lines / total_lines
        
    def step(self):
        """Time step update"""
        self.update_knowledge_effect()
        
    def calculate_equipment_efficiency(self):
        """Calculate current equipment utilization efficiency
        
        Returns:
            float: Current equipment efficiency value
        """
        return self.equipment_efficiency
        
    def update_metrics(self, completed_order):
        """Update order completion metrics"""
        # Update delay-related metrics
        if completed_order.is_delayed:
            self.delayed_orders += 1
            self.total_delay_time += completed_order.delay_time
        
        self.completed_orders += 1  # Only increment when order is fully completed
        
        # Update cost and profit
        # cdes_t = v_j × era_j × (u_j+1) × k_t
        design_cost = (completed_order.value * completed_order.era_ratio * 
                      (completed_order.design_mismatch + 1) * self.knowledge_effect)
        
        # Only add design cost, production cost is handled in update_production_cost
        self.total_cost += design_cost
        self.total_profit += (completed_order.value - design_cost)
        
    def get_average_delay_time(self) -> float:
        """Calculate average delay time"""
        if self.completed_orders == 0:
            return 0.0
        return self.total_delay_time / self.completed_orders

    def update_production_cost(self, current_cost: float):
        """Update production cost"""
        self.total_cost += current_cost