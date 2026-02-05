import os
import sys
import logging

# Get log level from environment variable, default to INFO
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Get distributed training rank (0 if not in distributed mode)
RANK = int(os.getenv("RANK", 0))

# Check if colors should be disabled (for piped output or NO_COLOR env var)
NO_COLOR = os.getenv("NO_COLOR") or not sys.stderr.isatty()

# Only configure logging for rank 0 in distributed training
_logging_configured = False


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    def __init__(self, use_colors=True):
        self.use_colors = use_colors and not NO_COLOR
        # Format: timestamp - module - LEVEL - message
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def format(self, record):
        if self.use_colors:
            # Shorten module name for readability (e.g., matcha.dataset.pdbbind -> dataset.pdbbind)
            if record.name.startswith('matcha.'):
                record.name = record.name[9:]  # Remove 'matcha.' prefix
            
            # Color the level name
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
            
            # Make timestamp dimmer for less visual noise
            formatted = super().format(record)
            # Add dim color to timestamp (first 19 chars: YYYY-MM-DD HH:MM:SS)
            formatted = f"{self.DIM}{formatted[:19]}{self.RESET}{formatted[19:]}"
            
            return formatted
        else:
            # No colors - just shorten module names
            if record.name.startswith('matcha.'):
                record.name = record.name[9:]
            return super().format(record)


def setup_logging():
    """Setup logging configuration. Should be called once at the start of the program."""
    global _logging_configured
    if _logging_configured:
        return
    
    # Only log from rank 0 in distributed training
    if RANK != 0:
        # Disable all logging for non-zero ranks
        logging.disable(logging.CRITICAL)
        _logging_configured = True
        return
    
    # Create console handler with colored formatter
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(ColoredFormatter(use_colors=True))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    root_logger.handlers.clear()  # Remove any existing handlers
    root_logger.addHandler(console_handler)
    
    # Disable third-party library loggers to reduce noise
    logging.getLogger('prody').setLevel(logging.CRITICAL)
    logging.getLogger('.prody').setLevel(logging.CRITICAL)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('posebusters.posebusters').setLevel(logging.WARNING)
    
    # Silence wandb and git (wandb auto-initializes when installed, even if not used)
    logging.getLogger('wandb').setLevel(logging.WARNING)
    logging.getLogger('wandb.docker').setLevel(logging.WARNING)
    logging.getLogger('wandb.docker.auth').setLevel(logging.CRITICAL)
    logging.getLogger('git').setLevel(logging.WARNING)
    logging.getLogger('git.cmd').setLevel(logging.CRITICAL)
    
    _logging_configured = True


def get_logger(name):
    """
    Get a logger instance for the given module name.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    if not _logging_configured:
        setup_logging()
    return logging.getLogger(name)


# Setup logging when this module is imported
setup_logging()
