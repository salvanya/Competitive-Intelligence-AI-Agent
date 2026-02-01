"""
Rate Limiter for Gemini API Calls
Token bucket algorithm implementation for 15 RPM free tier compliance
"""

import time
import asyncio
from typing import List


class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    
    Enforces rate limits by tracking timestamps of recent API calls and
    automatically waiting when the limit is exceeded. Uses a sliding window
    approach to ensure compliance with per-minute rate limits.
    
    Attributes:
        calls_per_minute: Maximum number of API calls allowed per minute
        calls: List of timestamps for recent API calls (within last 60 seconds)
    
    Example:
        >>> limiter = RateLimiter(calls_per_minute=15)
        >>> await limiter.acquire()  # Waits if necessary, then proceeds
        >>> # Make API call here
    """
    
    def __init__(self, calls_per_minute: int = 15):
        """
        Initialize the rate limiter.
        
        Args:
            calls_per_minute: Maximum number of calls allowed per minute
                             (default: 15 for Gemini free tier)
        
        Raises:
            ValueError: If calls_per_minute is less than 1
        """
        if calls_per_minute < 1:
            raise ValueError("calls_per_minute must be at least 1")
        
        self.calls_per_minute = calls_per_minute
        self.calls: List[float] = []
    
    async def acquire(self) -> None:
        """
        Acquire permission to make an API call.
        
        This method checks if making a call would exceed the rate limit.
        If so, it calculates the required wait time and sleeps asynchronously.
        Once the rate limit allows, it records the call timestamp and returns.
        
        This is a blocking call that will wait as long as necessary to
        respect the rate limit.
        
        Returns:
            None
        
        Example:
            >>> limiter = RateLimiter(calls_per_minute=15)
            >>> await limiter.acquire()
            >>> response = await api_client.call()  # Safe to proceed
        """
        now = time.time()
        
        # Remove calls older than 60 seconds (sliding window)
        self.calls = [timestamp for timestamp in self.calls if now - timestamp < 60]
        
        # Check if we've hit the rate limit
        if len(self.calls) >= self.calls_per_minute:
            # Calculate wait time based on oldest call in the window
            oldest_call = min(self.calls)
            wait_time = 60 - (now - oldest_call) + 1  # +1 second buffer for safety
            
            if wait_time > 0:
                # Wait and then recursively check again
                await asyncio.sleep(wait_time)
                return await self.acquire()
        
        # Record this call and allow it to proceed
        self.calls.append(now)
    
    def get_stats(self) -> dict:
        """
        Get current rate limit statistics.
        
        Useful for monitoring and debugging rate limit behavior.
        Shows how many calls have been made in the last minute and
        how much capacity remains.
        
        Returns:
            dict: Dictionary containing:
                - calls_last_minute: Number of calls in the last 60 seconds
                - remaining_capacity: Number of calls still allowed
                - calls_per_minute: Maximum calls allowed per minute
        
        Example:
            >>> stats = limiter.get_stats()
            >>> print(f"Used: {stats['calls_last_minute']}/{stats['calls_per_minute']}")
            Used: 12/15
        """
        now = time.time()
        recent_calls = [timestamp for timestamp in self.calls if now - timestamp < 60]
        
        return {
            "calls_last_minute": len(recent_calls),
            "remaining_capacity": self.calls_per_minute - len(recent_calls),
            "calls_per_minute": self.calls_per_minute,
        }
    
    def reset(self) -> None:
        """
        Reset the rate limiter by clearing all call history.
        
        Useful for testing or when starting a new session.
        
        Returns:
            None
        """
        self.calls.clear()
    
    def __repr__(self) -> str:
        """String representation of the rate limiter."""
        stats = self.get_stats()
        return (
            f"RateLimiter(calls_per_minute={self.calls_per_minute}, "
            f"used={stats['calls_last_minute']}, "
            f"remaining={stats['remaining_capacity']})"
        )