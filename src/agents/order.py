# src/agents/order.py

class Order:
    """Order agent class"""
    
    __slots__ = [
        'id', 'generation_time', 'duration', 'value', 'quality_req', 'era_ratio',
        'expected_design_duration', 'expected_production_duration',
        'expected_completion_time', 'design_start_time', 'design_completion_time',
        'production_start_time', 'completion_time', '_delay_time', 'is_completed',
        'design_mismatch', 'assigned_line'
    ]
    
    def __init__(self,
                 id: int,                  # order_id
                 generation_time: int,      # ge_j
                 duration: int,            # du_j
                 value: float,             # v_j
                 quality_req: int,         # qu_j
                 era_ratio: float = 0.25):  # era_j  

        # Basic attributes
        self.id = id
        self.generation_time = generation_time
        self.duration = duration
        self.value = value
        self.quality_req = quality_req
        self.era_ratio = era_ratio
        
        # Time-related attributes
        self.expected_design_duration = int(duration * era_ratio)
        self.expected_production_duration = duration - self.expected_design_duration
        self.expected_completion_time = generation_time + duration
        
        # Actual time records
        self.design_start_time = None
        self.design_completion_time = None
        self.production_start_time = None
        self.completion_time = None
        
        # Status variables
        self._delay_time = 0
        self.is_completed = False
        self.design_mismatch = 0
        self.assigned_line = None
    
    @property
    def delay_time(self) -> int:
        """Get delay time"""
        if not self.is_completed or self.completion_time is None:
            return 0
        return max(0, self.completion_time - self.expected_completion_time)
    
    @property
    def total_time(self) -> int:
        """Get total processing time"""
        if not self.is_completed:
            return 0
        return self.completion_time - self.generation_time
    
    @property
    def design_time(self) -> int:
        """Get design phase time"""
        if self.design_completion_time is None:
            return 0
        return self.design_completion_time - self.design_start_time
    
    @property
    def production_time(self) -> int:
        """Get production phase time"""
        if self.completion_time is None:
            return 0
        return self.completion_time - self.production_start_time
    
    @property
    def is_delayed(self) -> bool:
        """Check if order is delayed"""
        return self.delay_time > 0
    
    def calculate_design_cost(self, quality_mismatch: float) -> float:
        """Calculate design cost"""
        return self.value * quality_mismatch