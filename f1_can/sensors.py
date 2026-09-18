from dataclasses import dataclass
from enum import Enum

class FaultType(Enum):
    RPM_SPIKE = "rpm_spike"
    SPEED_OFFSET = "speed_offset"
    THROTTLE_STUCK = "throttle_stuck"
    GEAR = "gear"

# The @dataclass is a decorator which will automatically generate certain methods for classes(like the constructor) 
@dataclass 
class SensorInfo:
    name: str
    max_val: float
    fault: FaultType
    index: int
class Sensors:
    RPM = SensorInfo(name="RPM", max_val=16000, fault=FaultType.RPM_SPIKE, index = 0)
    SPEED = SensorInfo(name="Speed", max_val=400, fault=FaultType.SPEED_OFFSET, index = 1)
    THROTTLE = SensorInfo(name="Throttle", max_val=100, fault=FaultType.THROTTLE_STUCK,index = 2)
    GEAR = SensorInfo(name="nGear", max_val=8, fault=FaultType.GEAR, index = 3)

    ALL_SENSORS = [RPM, SPEED, THROTTLE, GEAR]
    SENSOR_NAME_COLUMNS = [i.name for i in ALL_SENSORS]
    FAULT_TYPES = [i.fault for i in ALL_SENSORS]
    