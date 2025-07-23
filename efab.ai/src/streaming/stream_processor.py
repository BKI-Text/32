#!/usr/bin/env python3
"""
Real-time Streaming Data Processor
Beverly Knits AI Supply Chain Planner
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty
import time

logger = logging.getLogger(__name__)

@dataclass
class StreamEvent:
    """Represents a streaming event"""
    event_id: str
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime
    source: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source
        }

class StreamBuffer:
    """Buffer for streaming events with windowing capabilities"""
    
    def __init__(self, max_size: int = 1000, window_duration: timedelta = timedelta(minutes=5)):
        self.max_size = max_size
        self.window_duration = window_duration
        self.events: List[StreamEvent] = []
        self.lock = threading.Lock()
        
    def add_event(self, event: StreamEvent):
        """Add event to buffer"""
        with self.lock:
            self.events.append(event)
            
            # Remove old events
            cutoff_time = datetime.now() - self.window_duration
            self.events = [e for e in self.events if e.timestamp > cutoff_time]
            
            # Limit buffer size
            if len(self.events) > self.max_size:
                self.events = self.events[-self.max_size:]
                
    def get_events(self, event_type: Optional[str] = None, 
                   since: Optional[datetime] = None) -> List[StreamEvent]:
        """Get events from buffer with optional filtering"""
        with self.lock:
            filtered_events = self.events.copy()
            
            if event_type:
                filtered_events = [e for e in filtered_events if e.event_type == event_type]
                
            if since:
                filtered_events = [e for e in filtered_events if e.timestamp > since]
                
            return filtered_events
            
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self.lock:
            if not self.events:
                return {'count': 0, 'types': {}, 'oldest': None, 'newest': None}
                
            event_types = {}
            for event in self.events:
                event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
                
            return {
                'count': len(self.events),
                'types': event_types,
                'oldest': self.events[0].timestamp.isoformat() if self.events else None,
                'newest': self.events[-1].timestamp.isoformat() if self.events else None
            }

class RealTimeProcessor:
    """Real-time data processor with ML integration"""
    
    def __init__(self):
        self.buffer = StreamBuffer()
        self.processors: Dict[str, Callable] = {}
        self.running = False
        self.processing_queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def register_processor(self, event_type: str, processor: Callable):
        """Register a processor for specific event type"""
        self.processors[event_type] = processor
        logger.info(f"Registered processor for event type: {event_type}")
        
    async def process_event(self, event: StreamEvent) -> Dict[str, Any]:
        """Process a single event"""
        try:
            # Add to buffer
            self.buffer.add_event(event)
            
            # Process with registered processor
            processor = self.processors.get(event.event_type)
            if processor:
                result = await self._run_processor(processor, event)
                logger.debug(f"Processed event {event.event_id}: {result}")
                return result
            else:
                logger.warning(f"No processor registered for event type: {event.event_type}")
                return {'status': 'no_processor', 'event_id': event.event_id}
                
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}")
            return {'status': 'error', 'event_id': event.event_id, 'error': str(e)}
            
    async def _run_processor(self, processor: Callable, event: StreamEvent) -> Dict[str, Any]:
        """Run processor in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, processor, event)
        
    async def start_processing(self):
        """Start the real-time processing loop"""
        self.running = True
        logger.info("Started real-time processing")
        
        while self.running:
            try:
                # Check for new events in queue
                if not self.processing_queue.empty():
                    try:
                        event_data = self.processing_queue.get_nowait()
                        event = StreamEvent(**event_data)
                        await self.process_event(event)
                    except Empty:
                        pass
                        
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1)
                
    def stop_processing(self):
        """Stop the processing loop"""
        self.running = False
        logger.info("Stopped real-time processing")
        
    def add_event_to_queue(self, event_data: Dict[str, Any]):
        """Add event to processing queue"""
        self.processing_queue.put(event_data)
        
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        return self.buffer.get_stats()
        
    def get_recent_events(self, event_type: Optional[str] = None, 
                         minutes: int = 5) -> List[Dict[str, Any]]:
        """Get recent events"""
        since = datetime.now() - timedelta(minutes=minutes)
        events = self.buffer.get_events(event_type=event_type, since=since)
        return [event.to_dict() for event in events]

class MLStreamProcessor:
    """ML-specific streaming processor"""
    
    def __init__(self):
        self.processor = RealTimeProcessor()
        self.ml_results = {}
        
    async def start(self):
        """Start ML streaming processor"""
        # Register ML processors
        self.processor.register_processor('demand_update', self.process_demand_update)
        self.processor.register_processor('inventory_update', self.process_inventory_update)
        self.processor.register_processor('price_update', self.process_price_update)
        self.processor.register_processor('supplier_update', self.process_supplier_update)
        self.processor.register_processor('forecast_request', self.process_forecast_request)
        
        # Start processing
        await self.processor.start_processing()
        
    async def stop(self):
        """Stop ML streaming processor"""
        self.processor.stop_processing()
        
    def process_demand_update(self, event: StreamEvent) -> Dict[str, Any]:
        """Process demand update event"""
        try:
            data = event.data
            material_id = data.get('material_id')
            demand_value = data.get('demand', 0)
            
            # Simple demand analysis
            recent_events = self.processor.buffer.get_events(
                event_type='demand_update',
                since=datetime.now() - timedelta(hours=1)
            )
            
            recent_demands = [e.data.get('demand', 0) for e in recent_events 
                             if e.data.get('material_id') == material_id]
            
            if recent_demands:
                avg_demand = np.mean(recent_demands)
                demand_trend = 'increasing' if demand_value > avg_demand else 'decreasing'
                volatility = np.std(recent_demands) if len(recent_demands) > 1 else 0
            else:
                avg_demand = demand_value
                demand_trend = 'stable'
                volatility = 0
                
            result = {
                'material_id': material_id,
                'current_demand': demand_value,
                'average_demand': avg_demand,
                'trend': demand_trend,
                'volatility': volatility,
                'alert': volatility > avg_demand * 0.3  # Alert if high volatility
            }
            
            self.ml_results[f"demand_{material_id}"] = result
            return result
            
        except Exception as e:
            logger.error(f"Error processing demand update: {e}")
            return {'status': 'error', 'error': str(e)}
            
    def process_inventory_update(self, event: StreamEvent) -> Dict[str, Any]:
        """Process inventory update event"""
        try:
            data = event.data
            material_id = data.get('material_id')
            inventory_level = data.get('inventory_level', 0)
            
            # Simple inventory analysis
            if inventory_level < 100:  # Low inventory threshold
                urgency = 'high'
                recommendation = 'immediate_reorder'
            elif inventory_level < 500:
                urgency = 'medium'
                recommendation = 'schedule_reorder'
            else:
                urgency = 'low'
                recommendation = 'monitor'
                
            result = {
                'material_id': material_id,
                'inventory_level': inventory_level,
                'urgency': urgency,
                'recommendation': recommendation,
                'timestamp': event.timestamp.isoformat()
            }
            
            self.ml_results[f"inventory_{material_id}"] = result
            return result
            
        except Exception as e:
            logger.error(f"Error processing inventory update: {e}")
            return {'status': 'error', 'error': str(e)}
            
    def process_price_update(self, event: StreamEvent) -> Dict[str, Any]:
        """Process price update event"""
        try:
            data = event.data
            material_id = data.get('material_id')
            price = data.get('price', 0)
            
            # Simple price analysis
            recent_events = self.processor.buffer.get_events(
                event_type='price_update',
                since=datetime.now() - timedelta(days=1)
            )
            
            recent_prices = [e.data.get('price', 0) for e in recent_events 
                            if e.data.get('material_id') == material_id]
            
            if recent_prices:
                avg_price = np.mean(recent_prices)
                price_change = ((price - avg_price) / avg_price) * 100 if avg_price > 0 else 0
            else:
                avg_price = price
                price_change = 0
                
            result = {
                'material_id': material_id,
                'current_price': price,
                'average_price': avg_price,
                'price_change_percent': price_change,
                'alert': abs(price_change) > 10  # Alert if price change > 10%
            }
            
            self.ml_results[f"price_{material_id}"] = result
            return result
            
        except Exception as e:
            logger.error(f"Error processing price update: {e}")
            return {'status': 'error', 'error': str(e)}
            
    def process_supplier_update(self, event: StreamEvent) -> Dict[str, Any]:
        """Process supplier update event"""
        try:
            data = event.data
            supplier_id = data.get('supplier_id')
            
            # Simple supplier analysis
            result = {
                'supplier_id': supplier_id,
                'status': data.get('status', 'unknown'),
                'reliability_score': data.get('reliability_score', 0.8),
                'last_update': event.timestamp.isoformat()
            }
            
            self.ml_results[f"supplier_{supplier_id}"] = result
            return result
            
        except Exception as e:
            logger.error(f"Error processing supplier update: {e}")
            return {'status': 'error', 'error': str(e)}
            
    def process_forecast_request(self, event: StreamEvent) -> Dict[str, Any]:
        """Process forecast request event"""
        try:
            data = event.data
            material_id = data.get('material_id')
            periods = data.get('periods', 7)
            
            # Simple forecast generation
            recent_events = self.processor.buffer.get_events(
                event_type='demand_update',
                since=datetime.now() - timedelta(days=30)
            )
            
            recent_demands = [e.data.get('demand', 0) for e in recent_events 
                             if e.data.get('material_id') == material_id]
            
            if recent_demands:
                avg_demand = np.mean(recent_demands)
                trend = (recent_demands[-1] - recent_demands[0]) / len(recent_demands) if len(recent_demands) > 1 else 0
                
                # Generate simple forecast
                forecast = []
                for i in range(periods):
                    forecast_value = avg_demand + (trend * i)
                    forecast.append({
                        'period': i + 1,
                        'forecast': max(0, forecast_value),
                        'confidence': 0.7
                    })
            else:
                # Default forecast if no historical data
                forecast = [{'period': i + 1, 'forecast': 100, 'confidence': 0.5} 
                           for i in range(periods)]
                
            result = {
                'material_id': material_id,
                'forecast': forecast,
                'generated_at': event.timestamp.isoformat()
            }
            
            self.ml_results[f"forecast_{material_id}"] = result
            return result
            
        except Exception as e:
            logger.error(f"Error processing forecast request: {e}")
            return {'status': 'error', 'error': str(e)}
            
    def get_ml_results(self) -> Dict[str, Any]:
        """Get latest ML results"""
        return self.ml_results.copy()
        
    def add_streaming_event(self, event_type: str, data: Dict[str, Any], source: str = 'api'):
        """Add streaming event for processing"""
        event_data = {
            'event_id': f"{event_type}_{int(time.time() * 1000)}",
            'event_type': event_type,
            'data': data,
            'timestamp': datetime.now(),
            'source': source
        }
        
        self.processor.add_event_to_queue(event_data)
        
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        return {
            'buffer_stats': self.processor.get_buffer_stats(),
            'ml_results_count': len(self.ml_results),
            'processor_running': self.processor.running
        }

# Global streaming processor instance
ml_stream_processor = MLStreamProcessor()

async def start_streaming_processor():
    """Start the global streaming processor"""
    logger.info("Starting ML streaming processor...")
    await ml_stream_processor.start()

async def stop_streaming_processor():
    """Stop the global streaming processor"""
    logger.info("Stopping ML streaming processor...")
    await ml_stream_processor.stop()

def get_streaming_processor() -> MLStreamProcessor:
    """Get the global streaming processor instance"""
    return ml_stream_processor