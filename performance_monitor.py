import psutil
import GPUtil
import logging
from datetime import datetime
import threading
import torch
import time

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self, config):
        self.config = config
        self._monitoring = False
        self._monitor_thread = None
        self._log_to_file = False  # Nouveau flag pour contrôler le logging dans le fichier
        
    def start(self, log_to_file=False):
        """Démarre le monitoring avec option de log dans fichier"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._log_to_file = log_to_file
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
    def stop(self):
        """Arrête le monitoring"""
        self._monitoring = False
        self._log_to_file = False
        if self._monitor_thread:
            self._monitor_thread.join()
            
    def _monitor_loop(self):
        while self._monitoring:
            try:
                # CPU Usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # RAM Usage
                ram = psutil.virtual_memory()
                ram_used_gb = ram.used / (1024**3)
                ram_total_gb = ram.total / (1024**3)
                
                # GPU Usage
                gpu_stats = []
                if torch.cuda.is_available():
                    for gpu in GPUtil.getGPUs():
                        gpu_stats.append({
                            'id': gpu.id,
                            'name': gpu.name,
                            'memory_used': gpu.memoryUsed,
                            'memory_total': gpu.memoryTotal,
                            'gpu_util': gpu.load * 100
                        })
                
                # Préparer le message de log
                log_message = (
                    f"Performance Metrics:\n"
                    f"CPU Usage: {cpu_percent}%\n"
                    f"RAM Usage: {ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB ({ram.percent}%)"
                )
                
                if gpu_stats:
                    for gpu in gpu_stats:
                        log_message += (
                            f"\nGPU {gpu['id']} ({gpu['name']}):\n"
                            f"Memory: {gpu['memory_used']}MB / {gpu['memory_total']}MB\n"
                            f"Utilization: {gpu['gpu_util']:.1f}%"
                        )

                # Logger uniquement dans la console si log_to_file est False
                if self._log_to_file:
                    logger.info(log_message)
                else:
                    print(log_message)
                
            except Exception as e:
                if self._log_to_file:
                    logger.error(f"Error in performance monitoring: {e}")
                
            time.sleep(self.config.PERFORMANCE_LOG_INTERVAL)