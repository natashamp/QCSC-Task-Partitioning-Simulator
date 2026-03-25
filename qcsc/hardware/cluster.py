"""Hardware cluster managing a collection of heterogeneous devices."""

from qcsc.config import DeviceType
from .device import Device, CPU, GPU, AIAccelerator, QPU


class HardwareCluster:
    """Collection of heterogeneous compute devices."""

    def __init__(self):
        self.devices: list[Device] = []
        self._by_type: dict[DeviceType, list[Device]] = {dt: [] for dt in DeviceType}

    def add_device(self, device: Device) -> None:
        self.devices.append(device)
        self._by_type[device.device_type].append(device)

    def get_devices_by_type(self, device_type: DeviceType) -> list[Device]:
        return self._by_type[device_type]

    def get_least_loaded(self, device_type: DeviceType) -> Device | None:
        devices = self._by_type[device_type]
        available = [d for d in devices if d.current_load < d.capacity]
        if not available:
            return None
        return min(available, key=lambda d: d.utilization)

    def get_all_available(self) -> list[Device]:
        return [d for d in self.devices if d.current_load < d.capacity]

    @staticmethod
    def default_cluster() -> "HardwareCluster":
        """Create a default heterogeneous cluster."""
        cluster = HardwareCluster()

        # 2 CPU nodes
        for i in range(2):
            cluster.add_device(CPU(id=f"cpu_{i}", name=f"CPU-{i}"))

        # 2 GPUs
        for i in range(2):
            cluster.add_device(GPU(id=f"gpu_{i}", name=f"GPU-{i}"))

        # 1 AI Accelerator
        cluster.add_device(AIAccelerator(id="ai_accel_0", name="AI-Accelerator-0"))

        # 1 QPU
        cluster.add_device(QPU(id="qpu_0", name="QPU-0", qubit_count=127))

        return cluster

    def summary(self) -> str:
        lines = ["Hardware Cluster:"]
        for dt in DeviceType:
            devices = self._by_type[dt]
            if devices:
                lines.append(f"  {dt.name}: {len(devices)} device(s)")
                for d in devices:
                    lines.append(
                        f"    {d.name}: capacity={d.capacity}, "
                        f"speed={d.speed_factor:.1f}x, "
                        f"load={d.current_load}/{d.capacity}"
                    )
        return "\n".join(lines)
