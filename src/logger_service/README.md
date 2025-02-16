# Logger Service

A singleton logger service that provides centralized logging functionality with file rotation and function decoration capabilities.

## Features

- Singleton pattern implementation
- Automatic log file rotation (1MB size limit, 5 backup files)
- Function decoration for automatic entry/exit logging
- Multiple logging levels (debug, info, warning, error, critical)
- Automatic module and function name detection

## Usage

### Basic Logging

```python
from logger_service.logger import LoggerService

# Initialize logger
logger = LoggerService()

# Use different logging levels
logger.debug("Debug message")
logger.info("Information message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")
```

### Function Decoration

You can automatically log function entry and exit using the decorator:

```python
from logger_service.logger import LoggerService

@LoggerService.log_function(level='info')
def my_function():
    # Your code here
    pass

@LoggerService.log_function(level='debug')
def another_function():
    # Your code here
    pass
```

### Log Output Format

Logs are written to `logs/app.log` with the following format:
```
timestamp - level - module - function - message
```

### Log File Location

Logs are stored in the `logs` directory at the project root:
- Maximum file size: 1MB
- Maximum backup files: 5
- Backup files are named: app.log.1, app.log.2, etc.

## Example

```python
from logger_service.logger import LoggerService

logger = LoggerService()

class MyClass:
    @LoggerService.log_function()
    def some_method(self):
        logger.info("Processing something...")
        try:
            # Some operations
            result = 1 + 1
            logger.debug(f"Calculation result: {result}")
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
```
