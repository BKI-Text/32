#!/usr/bin/env python3
"""
WebSocket Client Test for Real-time ML Inference
Beverly Knits AI Supply Chain Planner
"""

import asyncio
import json
import logging
from datetime import datetime
import websockets
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebSocketMLClient:
    """Test client for WebSocket ML inference"""
    
    def __init__(self, url: str):
        self.url = url
        self.websocket = None
        
    async def connect(self):
        """Connect to WebSocket server"""
        try:
            self.websocket = await websockets.connect(self.url)
            logger.info(f"Connected to WebSocket at {self.url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
            
    async def disconnect(self):
        """Disconnect from WebSocket server"""
        if self.websocket:
            await self.websocket.close()
            logger.info("Disconnected from WebSocket")
            
    async def send_message(self, message: dict):
        """Send message to WebSocket server"""
        if self.websocket:
            await self.websocket.send(json.dumps(message))
            logger.info(f"Sent message: {message['type']}")
            
    async def receive_messages(self):
        """Receive messages from WebSocket server"""
        if self.websocket:
            try:
                async for message in self.websocket:
                    data = json.loads(message)
                    logger.info(f"Received: {data['type']}")
                    self.handle_message(data)
            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection closed by server")
            except Exception as e:
                logger.error(f"Error receiving messages: {e}")
                
    def handle_message(self, data: dict):
        """Handle received message"""
        message_type = data.get('type', 'unknown')
        
        if message_type == 'forecast_complete':
            logger.info(f"Forecast complete for {data.get('model_type', 'unknown')} model")
            predictions = data.get('predictions', [])
            logger.info(f"Received {len(predictions)} predictions")
            
        elif message_type == 'risk_assessment_complete':
            risk_score = data.get('risk_score', {})
            logger.info(f"Risk assessment complete: {risk_score.get('risk_level', 'unknown')} risk")
            
        elif message_type == 'system_status':
            models = data.get('models', {})
            logger.info(f"System status - Active models: {sum(models.values())}")
            
        elif message_type == 'error':
            logger.error(f"Server error: {data.get('message', 'Unknown error')}")
            
        else:
            logger.info(f"Unknown message type: {message_type}")
            
    async def test_forecast_request(self):
        """Test forecast request"""
        request = {
            'type': 'forecast',
            'model': 'prophet',
            'periods': 7,
            'data': [
                {'date': '2025-01-01', 'demand': 1000},
                {'date': '2025-01-02', 'demand': 1100},
                {'date': '2025-01-03', 'demand': 1200}
            ]
        }
        
        await self.send_message(request)
        
    async def test_risk_assessment_request(self):
        """Test risk assessment request"""
        request = {
            'type': 'risk_assessment',
            'supplier_data': {
                'supplier_id': 'SUP001',
                'reliability_score': 0.85,
                'lead_time_days': 14,
                'cost_factor': 1.1
            }
        }
        
        await self.send_message(request)
        
    async def test_status_request(self):
        """Test status request"""
        request = {
            'type': 'status'
        }
        
        await self.send_message(request)
        
    async def run_test_sequence(self):
        """Run a sequence of tests"""
        if not await self.connect():
            return False
            
        try:
            # Start receiving messages in background
            receive_task = asyncio.create_task(self.receive_messages())
            
            # Send test requests
            logger.info("Testing status request...")
            await self.test_status_request()
            await asyncio.sleep(2)
            
            logger.info("Testing forecast request...")
            await self.test_forecast_request()
            await asyncio.sleep(2)
            
            logger.info("Testing risk assessment request...")
            await self.test_risk_assessment_request()
            await asyncio.sleep(2)
            
            # Wait for responses
            await asyncio.sleep(5)
            
            # Cancel receive task
            receive_task.cancel()
            
            await self.disconnect()
            return True
            
        except Exception as e:
            logger.error(f"Test sequence failed: {e}")
            await self.disconnect()
            return False

async def test_websocket_endpoints():
    """Test WebSocket endpoints"""
    logger.info("🚀 Testing WebSocket ML Inference Endpoints")
    
    # Test ML inference endpoint
    ml_client = WebSocketMLClient("ws://localhost:8000/api/v1/ws/ml-inference")
    success = await ml_client.run_test_sequence()
    
    if success:
        logger.info("✅ WebSocket ML inference test completed successfully!")
        return True
    else:
        logger.error("❌ WebSocket ML inference test failed!")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_websocket_endpoints())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)