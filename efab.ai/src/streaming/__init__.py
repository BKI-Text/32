"""
Real-time Streaming Data Processing Module
Beverly Knits AI Supply Chain Planner
"""

from .stream_processor import (
    StreamEvent,
    StreamBuffer,
    RealTimeProcessor,
    MLStreamProcessor,
    ml_stream_processor,
    start_streaming_processor,
    stop_streaming_processor,
    get_streaming_processor
)

__all__ = [
    'StreamEvent',
    'StreamBuffer', 
    'RealTimeProcessor',
    'MLStreamProcessor',
    'ml_stream_processor',
    'start_streaming_processor',
    'stop_streaming_processor',
    'get_streaming_processor'
]