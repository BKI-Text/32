#!/usr/bin/env python3
"""
Test Real-time Streaming Data Processing
Beverly Knits AI Supply Chain Planner
"""

import asyncio
import sys
import os
import logging
from datetime import datetime
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.streaming import MLStreamProcessor, StreamEvent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_streaming_processor():
    """Test the streaming processor"""
    logger.info("🚀 Testing Real-time Streaming Data Processing")
    
    try:
        # Create streaming processor
        processor = MLStreamProcessor()
        
        # Start processing (in background)
        processing_task = asyncio.create_task(processor.start())
        
        # Wait for processor to start
        await asyncio.sleep(1)
        
        # Test demand update events
        logger.info("Adding demand update events...")
        processor.add_streaming_event('demand_update', {
            'material_id': 'YARN001',
            'demand': 1000
        })
        
        processor.add_streaming_event('demand_update', {
            'material_id': 'YARN001', 
            'demand': 1200
        })
        
        # Test inventory update events
        logger.info("Adding inventory update events...")
        processor.add_streaming_event('inventory_update', {
            'material_id': 'YARN001',
            'inventory_level': 50  # Low inventory
        })
        
        # Test price update events
        logger.info("Adding price update events...")
        processor.add_streaming_event('price_update', {
            'material_id': 'YARN001',
            'price': 15.50
        })
        
        # Test forecast request
        logger.info("Adding forecast request...")
        processor.add_streaming_event('forecast_request', {
            'material_id': 'YARN001',
            'periods': 7
        })
        
        # Wait for processing
        await asyncio.sleep(3)
        
        # Get results
        ml_results = processor.get_ml_results()
        stats = processor.get_streaming_stats()
        
        logger.info("✅ Streaming processor test completed!")
        logger.info(f"ML Results: {len(ml_results)} items")
        
        # Show results
        for key, result in ml_results.items():
            logger.info(f"Result {key}: {result}")
            
        logger.info(f"Streaming Stats: {stats}")
        
        # Stop processing
        await processor.stop()
        processing_task.cancel()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Streaming processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_streaming_processor())
    sys.exit(0 if success else 1)