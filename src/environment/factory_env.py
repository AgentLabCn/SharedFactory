# src/environment/factory_env.py

import numpy as np
from typing import List, Dict, Optional
from ..agents.shared_factory import SharedFactory
from ..agents.design_department import DesignDepartment
from ..agents.production_line import ProductionLine
from ..agents.order import Order
import os
from datetime import datetime
import time

class FactoryEnvironment:
    """Factory environment class"""
    
    def __init__(self, 
                 num_production_lines: int ,    # Production capacity = 10
                 simulation_duration: int ,     # Simulation duration = 120
                 order_generation_interval: int , # Order generation interval
                 base_order_value: float ,    # Order value
                 knowledge_transfer_rate: float ,   # Knowledge transfer rate
                 inconvenience_level: float ,    # Inconvenience cost coefficient = 0.02
                 order_duration=None):
        
        # Save parameters
        self.num_production_lines = num_production_lines
        # Initialize parameters
        self.current_time = 0
        self.simulation_duration = simulation_duration
        self.order_generation_interval = order_generation_interval
        self.base_order_value = base_order_value
        self.knowledge_transfer_rate = knowledge_transfer_rate
        self.inconvenience_level = inconvenience_level  # Save inconvenience cost coefficient
        self.order_duration = order_duration
        
        # Store state data for all time steps
        self.data = []
        
        # Initialize log file
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = open(f"{self.log_dir}/simulation_log_{self.timestamp}.txt", "w", encoding='utf-8')
        self.log(f"Simulation started at {self.timestamp}")
        self.log(f"Parameters: production_lines={num_production_lines}, " +
                 f"duration={simulation_duration}, order_interval={order_generation_interval}, " +
                 f"knowledge_rate={knowledge_transfer_rate}, inconvenience={inconvenience_level}")
        
        # Initialize agents
        self.shared_factory = SharedFactory(
            alpha1=0.5,
            alpha2=0.3,
            alpha3=0.2,
            knowledge_transfer_rate=knowledge_transfer_rate
        )
        
        self.design_department = DesignDepartment(
            shared_factory=self.shared_factory,
            knowledge_transfer_rate=knowledge_transfer_rate
        )
        # Set design department capacity equal to number of production lines
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

    def reset(self):
        """Reset environment state"""
        # Close current log file
        try:
            if hasattr(self, 'log_file') and self.log_file and not self.log_file.closed:
                self.log(f"\nSimulation ended at time step {self.current_time}")
                # Ensure all data is written to disk
                self.log_file.flush()
                os.fsync(self.log_file.fileno())  
                self.log_file.close()
                
                # Add sleep time to ensure log file is fully written   
                time.sleep(1)
        except Exception as e:
            print(f"Error closing log file during reset: {e}")
        
        # Reset time
        self.current_time = 0
        
        # Reset order management
        self.active_orders = {}
        self.completed_orders = {}
        self.order_counter = 0
        
        # Reset shared factory
        self.shared_factory = SharedFactory(
            alpha1=0.5,
            alpha2=0.3,
            alpha3=0.2,
            knowledge_transfer_rate=self.knowledge_transfer_rate
        )
        
        # Reset design department
        self.design_department = DesignDepartment(
            shared_factory=self.shared_factory,
            knowledge_transfer_rate=self.knowledge_transfer_rate
        )
        self.design_department.set_max_concurrent_designs(self.num_production_lines)
        
        # Reset production lines
        self.production_lines = [
            ProductionLine(
                line_id=i,
                quality_index=np.random.randint(1, 6),
                base_inconvenience=500,
                inconvenience_level=self.inconvenience_level
            ) for i in range(self.num_production_lines)
        ]
        
        # Reset data list
        self.data = []
        
        # Create new log file
        try:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = open(f"{self.log_dir}/simulation_log_{self.timestamp}.txt", "w", encoding='utf-8')
            self.log(f"Simulation started at {self.timestamp}")
            self.log(f"Parameters: production_lines={self.num_production_lines}, " +
                    f"duration={self.simulation_duration}, order_interval={self.order_generation_interval}, " +
                    f"knowledge_rate={self.knowledge_transfer_rate}, inconvenience={self.inconvenience_level}")
        except Exception as e:
            print(f"Error creating new log file during reset: {e}")
        
    def generate_order(self) -> Order:
        """Generate order"""
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
        """Environment step"""
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
        newly_available_lines = []  # Record production lines that just completed orders
        
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
                    newly_available_lines.append(i)  # Record newly available lines
        
        # Immediately assign new orders to newly available lines
        if newly_available_lines:
            # Find all orders with completed designs but no assigned production line
            unassigned_orders = [order for order in self.active_orders.values() 
                                if order.design_completion_time is not None and 
                                order.assigned_line is None]
            
            # Sort by expected completion time (prioritize more urgent orders)
            unassigned_orders.sort(key=lambda x: x.expected_completion_time)
            
            # Assign orders to newly available lines
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
        state = self._get_current_state()
        
        # Save state to data list
        if state is not None:
            self.data.append(state)
            
        return state
        
    def _get_current_state(self):
        """Get current state"""
        total_orders = len(self.completed_orders) + len(self.active_orders)
        if total_orders == 0:
            self.log("No orders in system")
            return None
            
        # Calculate delay ratio - modified logic
        delayed_orders = []
        
        # 1. Completed and delayed orders
        completed_delayed = [order for order in self.completed_orders.values() 
                           if order.delay_time > 0]
        delayed_orders.extend(completed_delayed)
        
        # 2. Uncompleted orders past expected completion time
        active_delayed = [order for order in self.active_orders.values()
                         if not order.is_completed and 
                         self.current_time > order.expected_completion_time]
        delayed_orders.extend(active_delayed)
        
        # 3. Orders that will definitely be delayed based on production schedule
        expected_delayed = []
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
                    expected_delayed.append(order)
        delayed_orders.extend(expected_delayed)
        
        # Calculate delay ratio
        delay_ratio = len(delayed_orders) / max(1, total_orders)
        
        # Log delay information at last time step or when there are delays
        if self.current_time == self.simulation_duration - 1 or delay_ratio > 0:
            self.log(f"\n======== DELAY STATISTICS (Time Step {self.current_time}) ========")
            self.log(f"Total Orders: {total_orders}")
            self.log(f"Total Delayed Orders: {len(delayed_orders)}")
            self.log(f"Completed Delayed Orders: {len(completed_delayed)}")
            self.log(f"Active Overdue Orders: {len(active_delayed)}")
            self.log(f"Expected Delayed Orders: {len(expected_delayed)}")
            self.log(f"Delay Ratio: {delay_ratio:.4f}")
            
            if delayed_orders:
                self.log("\nDelayed Order Details:")
                for i, order in enumerate(delayed_orders):
                    delay_time = order.delay_time if order.is_completed else max(0, self.current_time - order.expected_completion_time)
                    status = "Completed" if order.is_completed else "In Progress"
                    self.log(f"  {i+1}. Order {order.id}: {status}, Delay Time: {delay_time}, Expected Completion Time: {order.expected_completion_time}")
            self.log("=======================================\n")
        
        # Calculate average delay time
        current_delays = []
        for order in delayed_orders:
            if order.is_completed:
                current_delays.append(order.delay_time)
            else:
                # For uncompleted orders, calculate current known delay time
                current_delay = max(0, self.current_time - order.expected_completion_time)
                if current_delay > 0:
                    current_delays.append(current_delay)
        
        avg_delay_time = (sum(current_delays) / max(1, len(current_delays))) if current_delays else 0
        
        # Calculate average processing time
        avg_total_time = (sum(order.total_time for order in self.completed_orders.values()) /
                         max(1, len(self.completed_orders)))
        
        # Calculate design and production times
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
                    # Find expected completion time for current order
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
        if hasattr(self, 'log_file') and self.log_file and not self.log_file.closed:
            self.log_file.write(message + "\n")
            self.log_file.flush()  # Ensure immediate write to file

    def __del__(self):
        """Destructor to ensure proper log file closure"""
        try:
            if hasattr(self, 'log_file') and self.log_file and not self.log_file.closed:
                self.log(f"\nSimulation ended at time step {self.current_time}")
                # Ensure all data is written to disk
                self.log_file.flush()
                os.fsync(self.log_file.fileno()) 
                self.log_file.close()
        except Exception as e:
            print(f"Error closing log file: {e}")

    def export_state_metrics(self, filename="detailed_metrics.xlsx"):
        """Export current environment state metrics to Excel file
        
        Args:
            filename (str): Excel filename to save
        """
        try:
            import pandas as pd
            
            if not self.data:
                self.log("Warning: No data to export")
                return
                
            # Convert current environment state data to DataFrame
            df = pd.DataFrame(self.data)
            
            # Create Excel file
            with pd.ExcelWriter(filename) as writer:
                df.to_excel(writer, sheet_name='state_metrics')
                
                # Add environment configuration information
                config_data = {
                    'num_production_lines': self.num_production_lines,
                    'simulation_duration': self.simulation_duration,
                    'order_generation_interval': self.order_generation_interval,
                    'knowledge_transfer_rate': self.knowledge_transfer_rate,
                    'inconvenience_level': self.inconvenience_level,
                    'order_duration': self.order_duration,
                }
                pd.DataFrame([config_data]).to_excel(writer, sheet_name='configuration')
                
                # Add order information
                if self.completed_orders:
                    order_data = []
                    for order_id, order in self.completed_orders.items():
                        order_data.append({
                            'order_id': order.id,
                            'generation_time': order.generation_time,
                            'duration': order.duration,
                            'value': order.value,
                            'quality_req': order.quality_req,
                            'design_time': order.design_time,
                            'production_time': order.production_time,
                            'total_time': order.total_time,
                            'is_delayed': order.is_delayed,
                            'delay_time': order.delay_time,
                        })
                    pd.DataFrame(order_data).to_excel(writer, sheet_name='completed_orders')
                
            self.log(f"Exported state metrics to {filename}")
            return True
            
        except Exception as e:
            import traceback
            self.log(f"Error exporting metrics: {e}")
            traceback.print_exc()
            return False
