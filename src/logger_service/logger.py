import logging
import os
import inspect
from logging.handlers import RotatingFileHandler
from functools import wraps
from typing import Callable
from configuration_manager.config_manager import ConfigManager

class LoggerService:
    _instance = None
    _handler = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerService, cls).__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance
    
    def _initialize_logger(self):
        # Configure root logger
        root_logger = logging.getLogger()
        config = ConfigManager()
        log_config = config.logging_config
        
        root_logger.setLevel(getattr(logging, log_config['log_level']))
        
        # Create and configure handler if not exists
        if not LoggerService._handler:
            LoggerService._handler = RotatingFileHandler(
                log_config['log_file'],
                maxBytes=log_config['max_bytes'],
                backupCount=log_config['backup_count']
            )
            
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s'
            )
            LoggerService._handler.setFormatter(formatter)
            root_logger.addHandler(LoggerService._handler)
    
    def _get_logger(self, caller_frame):
        # Get module name from caller
        module = inspect.getmodule(caller_frame)
        module_name = module.__name__ if module else 'unknown'
        if module_name == '__main__':
            module_name = os.path.splitext(os.path.basename(caller_frame.f_code.co_filename))[0]
        
        # Get or create logger for module
        logger = logging.getLogger(module_name)
        return logger
    
    def _log(self, level: str, message: str):
        # Find the caller frame
        stack = inspect.stack()
        caller = None
        for frame in stack[1:]:
            if 'logger_service' not in frame.filename:
                caller = frame
                break
        
        if caller:
            logger = self._get_logger(caller.frame)
            log_method = getattr(logger, level)
            log_method(message)
    
    def debug(self, message: str): self._log('debug', message)
    def info(self, message: str): self._log('info', message)
    def warning(self, message: str): self._log('warning', message)
    def error(self, message: str): self._log('error', message)
    def critical(self, message: str): self._log('critical', message)
    
    @staticmethod
    def log_function(level: str = 'info') -> Callable:
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                logger = LoggerService()
                # Get the module name from the decorated function
                module_name = inspect.getmodule(func).__name__
                if module_name == '__main__':
                    module_name = os.path.splitext(os.path.basename(inspect.getfile(func)))[0]
                
                logger._log(level, f"Entering {func.__name__}")
                try:
                    result = func(*args, **kwargs)
                    logger._log(level, f"Exiting {func.__name__}")
                    return result
                except Exception as e:
                    logger._log('error', f"Exception in {func.__name__}: {str(e)}")
                    raise
            return wrapper
        return decorator
