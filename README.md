# Factory Simulation README

## Factory Simulation Overview
Shared factories integrate manufacturing capacity and technological expertise to provide on-demand production services. This study evaluates critical operational factors—knowledge transfer rate, inconvenience level, order duration, and capacity—using an Agent-Based Model (ABM) simulating design, production, and order management processes. Through ANOVA analysis of simulation data, we demonstrate that capacity and order duration significantly impact all performance metrics (profit-cost ratio, equipment efficiency, and delay ratio). Our findings highlight strategic levers for optimizing shared factory operations, emphasizing capacity planning and contract duration alignment with technological learning rates. The ABM framework and experimental results provide actionable insights for practitioners in shared manufacturing systems.
![image](/base_case_results.png)
## Key Components
- **SharedFactory**: Manages the overall factory operations, resources, and knowledge effects.
- **DesignDepartment**: Handles order design using a queue system.
- **ProductionLine**: Processes orders with specific quality characteristics.
- **Order**: Represents customer orders, which have a duration, value, and requirements.
- **FactoryEnvironment**: The main environment that coordinates all agents and the simulation.
- **FactoryModel**: A wrapper around the environment for running simulations.

## Key Parameters
- **Production line capacity**: Options are 5, 10, or 15.
- **Knowledge transfer rate**: Can be 0.02, 0.04, or 0.06.
- **Inconvenience level**: Options are 0.01, 0.02, or 0.03.
- **Order duration**: Can be 24, 30, or 36.
- **Order generation interval**: Determines how often orders arrive.

## Performance Metrics
- **Profit - Cost Ratio**: Measures economic efficiency.
- **Equipment Efficiency**: Reflects production line utilization.
- **Delay Ratio**: The percentage of delayed orders.
- **Overall Performance Index**: A combined metric of the above.

## Running the Simulation
- **run_base_case.py**: Runs the simulation 30 times with default parameters and then averages the results.
- **run_experiment.py**: Runs factorial experiments with different parameter combinations.

## Process Flow
1. Orders are generated at regular intervals.
2. The design department processes orders up to its capacity.
3. Completed designs are assigned to production lines.
4. Production lines complete orders based on their capabilities.
5. Performance metrics are collected throughout the process.

## Knowledge Effect
The system has a learning mechanism. Accumulated experience (completed designs) reduces future design times. This is controlled by the knowledge transfer rate parameter.

## Analysis Tools
- The system includes ANOVA analysis capabilities.
- There are various visualization tools for analyzing results.
- Comprehensive logging is available for debugging.

## Purpose of the Simulation
The simulation is designed to study how different parameters affect factory performance. It specifically focuses on the trade - offs between efficiency, cost, and delivery timeliness in a shared factory environment.    
