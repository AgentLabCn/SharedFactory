# src/environment/factory_env.py

import numpy as np
from typing import List, Dict, Optional
from ..agents.shared_factory import SharedFactory
from ..agents.design_department import DesignDepartment
from ..agents.production_line import ProductionLine
from ..agents.order import Order
import os
from datetime import datetime

class FactoryEnvironment:
    """Factory environment class"""
    
    def __init__(self, 
                 num_production_lines: int,     # Production line capacity 
                 simulation_duration: int,      # Simulation duration 
                 order_generation_interval: int, # Order generation interval
                 base_order_value: float,       # Base order value
                 knowledge_transfer_rate: float, # Knowledge transfer rate
                 inconvenience_level: float,    # Inconvenience cost coefficient 
                 order_duration=None):
        
        # Save parameters
        self.num_production_lines = num_production_lines
        # Initialize parameters
        self.current_time = 0
        self.simulation_duration = simulation_duration
        self.order_generation_interval = order_generation_interval
        self.base_order_value = base_order_value
        self.knowledge_transfer_rate = knowledge_transfer_rate
        self.order_duration = order_duration
        
        # Initialize log file
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = open(f"{log_dir}/simulation_log_{timestamp}.txt", "w")
        self.log(f"Simulation started at {timestamp}")
        self.log(f"Parameters: production_lines={num_production_lines}, " +
                 f"duration={simulation_duration}, order_interval={order_generation_interval}, " +
                 f"knowledge_rate={knowledge_transfer_rate}, inconvenience={inconvenience_level}")
        
        # Initialize agents
        self.shared_factory = SharedFactory(
            alpha1=0.5,  # Cost-profit ratio weight
            alpha2=0.3,  # Equipment utilization weight
            alpha3=0.2,  # Delay rate weight
            knowledge_transfer_rate=knowledge_transfer_rate
        )
        
        self.design_department = DesignDepartment(
            shared_factory=self.shared_factory,
            knowledge_transfer_rate=knowledge_transfer_rate
        )
        # Set design department capacity equal to production lines
        self.design_department.set_max_concurrent_designs(num_production_lines)
        
        # Initialize production lines
        self.production_lines = [
            ProductionLine(
                line_id=i,
                quality_index=np.random.randint(1, 6),
                base_inconvenience=500,
                inconvenience_level=inconvenience_level
            ) for i in range(num_production_lines)
        ]
        
        # Initialize order management
        self.active_orders = {}
        self.completed_orders = {}
        self.order_counter = 0
        
    def generate_order(self) -> Order:
        """Generate new order"""
        if self.order_duration is not None:
            # Use specified duration
            duration = self.order_duration
        
        # Calculate order value
        value = self.base_order_value * (duration) * (1 + np.random.uniform(-0.1, 0.1)) 
        
        # Create order
        order = Order(
            id=self.order_counter,
            generation_time=self.current_time,
            duration=duration,
            value=value,
            quality_req=np.random.randint(1, 6),
            era_ratio=0.25
        )
        
        self.order_counter += 1
        return order
        
    def step(self):
        """Environment step update"""
        # Generate new order
        if self.current_time % self.order_generation_interval == 0:
            new_order = self.generate_order()
            self.active_orders[new_order.id] = new_order
            self.design_department.start_design(
                new_order, 
                self.current_time,
                self.shared_factory.knowledge_effect
            )
        
        # Update design department
        completed_designs = self.design_department.step(self.current_time)
        
        # Process completed designs
        for order_id in completed_designs:
            order = self.active_orders[order_id]
            best_line_id, design_mismatch = self.design_department.assign_order_to_line(
                order, self.production_lines, self.current_time)
            
            if best_line_id is not None:
                order.design_mismatch = design_mismatch
                order.assigned_line = best_line_id
                self.production_lines[best_line_id].start_production(
                    order, self.current_time)
        
        # Update production lines and calculate variable costs
        working_lines = 0
        current_production_cost = 0
        newly_available_lines = []  # Track lines that just completed orders
        
        for i, line in enumerate(self.production_lines):
            completed_order_id = line.step(self.current_time)
            
            if line.is_working:
                working_lines += 1
                current_production_cost += 50
            
            if completed_order_id is not None:
                completed_order = self.active_orders.get(completed_order_id)
                if completed_order:
                    completed_order.is_completed = True
                    self.shared_factory.update_metrics(completed_order)
                    self.completed_orders[completed_order_id] = self.active_orders.pop(completed_order_id)
                    newly_available_lines.append(i)  # Record newly available line
        
        # Immediately assign new orders to newly available lines
        if newly_available_lines:
            # Find all orders with completed designs but no assigned line
            unassigned_orders = [order for order in self.active_orders.values() 
                                if order.design_completion_time is not None and 
                                order.assigned_line is None]
            
            # Sort by expected completion time (prioritize more urgent orders)
            unassigned_orders.sort(key=lambda x: x.expected_completion_time)
            
            # Assign orders to each newly available line
            for line_id in newly_available_lines:
                if not unassigned_orders:
                    break  # No more orders to assign
                    
                order = unassigned_orders.pop(0)
                _, design_mismatch = self.design_department.assign_order_to_line(
                    order, [self.production_lines[line_id]], self.current_time)
                
                order.design_mismatch = design_mismatch
                order.assigned_line = line_id
                self.production_lines[line_id].start_production(
                    order, self.current_time)
                
                # Update working lines and cost (due to new assignment)
                working_lines += 1
                current_production_cost += 50
        
        # Update shared factory state
        self.shared_factory.update_production_cost(current_production_cost)
        self.shared_factory.update_equipment_efficiency(
            working_lines, len(self.production_lines))
        self.shared_factory.step()
        
        # Update time
        self.current_time += 1
        
        # Return current state
        return self._get_current_state()
        
    def _get_current_state(self):
        """Get current system state"""
        total_orders = len(self.completed_orders) + len(self.active_orders)
        if total_orders == 0:
            self.log("No orders in system")
            return None
            
        # Calculate delay ratio
        delayed_orders = []
        
        # 1. Completed and delayed orders
        delayed_orders.extend([order for order in self.completed_orders.values() 
                             if order.delay_time > 0])
        
        # 2. Uncompleted orders past expected completion time
        delayed_orders.extend([order for order in self.active_orders.values()
                             if not order.is_completed and 
                             self.current_time > order.expected_completion_time])
        
        # 3. Orders that will be delayed based on production schedule
        for order in self.active_orders.values():
            if not order.is_completed and order.assigned_line is not None:
                line = self.production_lines[order.assigned_line]
                # Find planned completion time
                planned_completion_time = None
                for time, order_id in line.production_schedule.items():
                    if order_id == order.id:
                        planned_completion_time = time
                        break
                
                if (planned_completion_time and 
                    planned_completion_time > order.expected_completion_time and
                    order not in delayed_orders):
                    delayed_orders.append(order)
        
        # Calculate delay ratio
        delay_ratio = len(delayed_orders) / max(1, total_orders)
        
        # Calculate average delay time
        current_delays = []
        for order in delayed_orders:
            if order.is_completed:
                current_delays.append(order.delay_time)
            else:
                # For uncompleted orders, calculate known delay time
                current_delay = max(0, self.current_time - order.expected_completion_time)
                if current_delay > 0:
                    current_delays.append(current_delay)
        
        avg_delay_time = (sum(current_delays) / max(1, len(current_delays))) if current_delays else 0
        
        # Calculate average processing time
        avg_total_time = (sum(order.total_time for order in self.completed_orders.values()) /
                         max(1, len(self.completed_orders)))
        
        # Calculate design and production time
        avg_design_time = (sum(order.design_time for order in self.completed_orders.values()) /
                          max(1, len(self.completed_orders)))
        avg_production_time = (sum(order.production_time for order in self.completed_orders.values()) /
                             max(1, len(self.completed_orders)))
        
        # Get other metrics
        knowledge_effect = max(0.1, self.shared_factory.knowledge_effect)
        equipment_efficiency = max(0.1, self.shared_factory.equipment_efficiency)
        
        # Calculate profit-cost ratio
        if self.shared_factory.total_cost > 0:
            profit_cost_ratio = self.shared_factory.total_profit / self.shared_factory.total_cost
        else:
            profit_cost_ratio = 0.0
        
        state = {
            'delay_ratio': delay_ratio,
            'knowledge_effect': knowledge_effect,
            'equipment_efficiency': equipment_efficiency,
            'profit_cost_ratio': profit_cost_ratio,
            'average_delay_time': avg_delay_time,
            'average_total_time': avg_total_time,
            'average_design_time': avg_design_time,
            'average_production_time': avg_production_time,
            'design_queue_length': self.design_department.get_queue_length(),
            'total_design_orders': self.design_department.get_total_orders()
        }
        
        # Output state
        if self.current_time % 1 == 0:
            self.log(f"\nTime step: {self.current_time}")
            self.log(f"Active orders: {len(self.active_orders)}")
            self.log(f"Completed orders: {len(self.completed_orders)}")
            self.log(f"Ongoing designs: {len(self.design_department.ongoing_designs)}")
            self.log(f"Design queue length: {len(self.design_department.design_queue)}")
            
            # Add design department details
            self.log("\nDesign Department Status:")
            for order_id, design_info in self.design_department.ongoing_designs.items():
                remaining_time = design_info['duration'] - design_info['progress']
                order = design_info['order']
                expected_completion = self.current_time + remaining_time
                self.log(f"  Order {order_id}: progress {design_info['progress']}/{design_info['duration']} " +
                      f"(remaining: {remaining_time}, expected completion: {expected_completion})")
                self.log(f"    Expected design duration: {order.expected_design_duration}, " +
                      f"Expected completion time: {order.expected_completion_time}")
            
            # Add production line details
            self.log("\nProduction Lines Status:")
            for i, line in enumerate(self.production_lines):
                if line.is_working and line.current_order:
                    # Find current order's expected completion time
                    next_completion_time = None
                    for time, order_id in line.production_schedule.items():
                        if order_id == line.current_order.id:
                            next_completion_time = time
                            break
                    
                    remaining_time = next_completion_time - self.current_time if next_completion_time else "unknown"
                    self.log(f"  Line {i}: Working on Order {line.current_order.id} " +
                          f"(remaining: {remaining_time}, completion at: {next_completion_time})")
                    self.log(f"    Order expected completion time: {line.current_order.expected_completion_time}, " +
                          f"Will be delayed: {next_completion_time > line.current_order.expected_completion_time}")
                else:
                    next_available = line.get_next_available_time()
                    if next_available > self.current_time:
                        self.log(f"  Line {i}: Scheduled to be available at {next_available}")
                    else:
                        self.log(f"  Line {i}: Available")
        
        return state
        
    def collect_metrics(self):
        """Collect key metrics"""
        return {
            'knowledge_effect': self.shared_factory.knowledge_effect,
            'cost_profit_ratio': self.shared_factory.total_profit / 
                                (self.shared_factory.total_cost + 1e-6),
            'equipment_efficiency': self.shared_factory.equipment_efficiency,
            'delay_ratio': self.shared_factory.delayed_orders / 
                          (len(self.shared_factory.accepted_orders) + 1e-6)
        }

    def log(self, message):
        """Write message to log file"""
        print(message)  # Also display in console
        self.log_file.write(message + "\n")
        self.log_file.flush()  # Ensure immediate write to file

    def __del__(self):
        """Destructor to ensure log file is properly closed"""
        if hasattr(self, 'log_file') and self.log_file:
            self.log(f"\nSimulation ended at time step {self.current_time}")
            self.log_file.close()