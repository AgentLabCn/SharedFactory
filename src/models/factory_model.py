# src/models/factory_model.py

import agentpy as ap
from typing import Dict
from src.environment.factory_env import FactoryEnvironment

# Factory model class for simulation

class FactoryModel(ap.Model):
    """Factory model class"""
    
    def setup(self):
        """Model initialization"""
        # Set parameters
        self.num_production_lines = self.p['num_production_lines']
        self.knowledge_transfer_rate = self.p['knowledge_transfer_rate']
        self.inconvenience_level = self.p['inconvenience_level']
        self.order_duration = self.p['order_duration']
        
        # Fixed parameters - Get fixed values from experiment
        self.simulation_duration = self.p['simulation_duration']
        self.order_generation_interval = self.p['order_generation_interval']
        self.base_order_value = self.p['base_order_value']
        
        # Create environment
        self.env = FactoryEnvironment(
            num_production_lines=self.num_production_lines,
            simulation_duration=self.simulation_duration,
            order_generation_interval=self.order_generation_interval,
            base_order_value=self.base_order_value, 
            order_duration=self.order_duration,
            knowledge_transfer_rate=self.knowledge_transfer_rate,
            inconvenience_level=self.inconvenience_level
        )
        
        # Data recording list
        self.data = []
        
    def step(self):
        """Model step update"""
        # Environment step and get state
        state = self.env.step()
        
        # Record state at each time step
        if state is not None:
            self.data.append(state)
        
        # Check if simulation should end
        if self.env.current_time >= self.simulation_duration:
            self.stop()
            
    def end(self) -> Dict:
        """Model end, return results"""
        if not self.data:
            print("Warning: No data collected during simulation")
            return {
                'final_state': {
                    'delay_ratio': 0.0,
                    'knowledge_effect': 0.0,
                    'equipment_efficiency': 0.0,
                    'profit_cost_ratio': 0.0
                }
            }
        
        final_state = self.data[-1]
        print("\nFinal simulation state:")
        print(f"Total orders completed: {len(self.env.completed_orders)}")
        print(f"Knowledge effect: {final_state['knowledge_effect']}")
        print(f"Equipment efficiency: {final_state['equipment_efficiency']}")
        print(f"Delay ratio: {final_state['delay_ratio']}")
        print(f"Profit/Cost ratio: {final_state['profit_cost_ratio']}")
        
        return {'final_state': final_state}