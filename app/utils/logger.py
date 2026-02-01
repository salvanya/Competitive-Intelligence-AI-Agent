"""
Progress Tracking and Logging Utilities
Provides structured logging and progress callback mechanisms
"""

import logging
from typing import Optional, Callable
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    """Log level enumeration for structured logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"


class ProgressLogger:
    """
    Progress tracking logger with callback support.
    
    Provides structured logging with optional callback functions for
    real-time progress updates (e.g., to Streamlit UI components).
    
    Attributes:
        name: Logger name (typically module or component name)
        callback: Optional callback function for progress updates
        logger: Python logging.Logger instance
    
    Example:
        >>> logger = ProgressLogger("scraper", callback=st.status_text.update)
        >>> logger.info("Starting scrape...")  # Updates Streamlit UI
    """
    
    def __init__(
        self, 
        name: str = "competitive-intel",
        callback: Optional[Callable[[str], None]] = None,
        level: int = logging.INFO
    ):
        """
        Initialize the progress logger.
        
        Args:
            name: Logger name (used for log identification)
            callback: Optional function to call with log messages
            level: Logging level (default: INFO)
        """
        self.name = name
        self.callback = callback
        
        # Set up Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Add console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _log_and_callback(self, level: LogLevel, message: str, emoji: str = "") -> None:
        """
        Internal method to log message and trigger callback.
        
        Args:
            level: Log level enum
            message: Log message
            emoji: Optional emoji prefix for callback
        """
        # Format message with emoji for callback
        formatted_message = f"{emoji} {message}" if emoji else message
        
        # Call callback if provided (for UI updates)
        if self.callback:
            try:
                self.callback(formatted_message)
            except Exception as e:
                self.logger.error(f"Callback error: {str(e)}")
        
        # Log to Python logger
        log_method = getattr(self.logger, level.value.lower(), self.logger.info)
        log_method(message)
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self._log_and_callback(LogLevel.DEBUG, message, "🔍")
    
    def info(self, message: str) -> None:
        """Log info message."""
        self._log_and_callback(LogLevel.INFO, message, "ℹ️")
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self._log_and_callback(LogLevel.WARNING, message, "⚠️")
    
    def error(self, message: str) -> None:
        """Log error message."""
        self._log_and_callback(LogLevel.ERROR, message, "❌")
    
    def success(self, message: str) -> None:
        """Log success message."""
        self._log_and_callback(LogLevel.SUCCESS, message, "✅")
    
    def progress(self, message: str, step: Optional[int] = None, total: Optional[int] = None) -> None:
        """
        Log progress update with optional step information.
        
        Args:
            message: Progress message
            step: Current step number (optional)
            total: Total number of steps (optional)
        
        Example:
            >>> logger.progress("Processing competitor", step=2, total=3)
            🔄 Processing competitor (2/3)
        """
        if step is not None and total is not None:
            formatted_message = f"{message} ({step}/{total})"
        else:
            formatted_message = message
        
        self._log_and_callback(LogLevel.INFO, formatted_message, "🔄")
    
    def section(self, title: str) -> None:
        """
        Log a section header for better visual organization.
        
        Args:
            title: Section title
        
        Example:
            >>> logger.section("Web Scraping Phase")
            ═══════════════════════════════════════
            📋 Web Scraping Phase
            ═══════════════════════════════════════
        """
        separator = "=" * 50
        header = f"\n{separator}\n📋 {title}\n{separator}"
        
        if self.callback:
            self.callback(f"📋 {title}")
        
        self.logger.info(header)
    
    def timestamp(self, message: str) -> None:
        """
        Log message with explicit timestamp.
        
        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamped_message = f"[{timestamp}] {message}"
        self._log_and_callback(LogLevel.INFO, timestamped_message, "🕒")
    
    def set_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """
        Update the callback function.
        
        Useful for changing UI update targets during execution.
        
        Args:
            callback: New callback function or None to disable
        """
        self.callback = callback


def create_logger(
    name: str = "competitive-intel",
    callback: Optional[Callable[[str], None]] = None
) -> ProgressLogger:
    """
    Factory function to create a ProgressLogger instance.
    
    Args:
        name: Logger name
        callback: Optional callback for progress updates
    
    Returns:
        ProgressLogger: Configured logger instance
    
    Example:
        >>> logger = create_logger("scraper", st.status.update)
        >>> logger.info("Starting process...")
    """
    return ProgressLogger(name=name, callback=callback)