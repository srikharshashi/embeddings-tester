from .logger import LoggerService

# Create a singleton instance
logger = LoggerService()

__all__ = ['logger', 'LoggerService']
