#!/usr/bin/env python3
"""
WebSocket Endpoints for Real-time ML Inference
Beverly Knits AI Supply Chain Planner
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from fastapi.routing import APIRouter
import pandas as pd
import numpy as np

from ..src.engine.ml_model_manager import MLModelManager
from ..src.engine.production_ml_loader import ProductionMLLoader
from ..src.monitoring.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

# Create WebSocket router
websocket_router = APIRouter()

class WebSocketManager:
    """Manager for WebSocket connections and real-time ML inference"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.ml_manager = MLModelManager()
        self.production_loader = ProductionMLLoader()
        self.performance_monitor = PerformanceMonitor()
        
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
        
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket"""
        await websocket.send_text(message)
        
    async def broadcast(self, message: str):
        """Send message to all connected WebSockets"""
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                
    async def process_ml_request(self, websocket: WebSocket, request_data: Dict[str, Any]):
        """Process real-time ML inference request"""
        try:
            request_type = request_data.get('type')
            
            if request_type == 'forecast':
                await self._handle_forecast_request(websocket, request_data)
            elif request_type == 'risk_assessment':
                await self._handle_risk_assessment_request(websocket, request_data)
            elif request_type == 'optimization':
                await self._handle_optimization_request(websocket, request_data)
            elif request_type == 'status':
                await self._handle_status_request(websocket, request_data)
            else:
                await self._send_error(websocket, f"Unknown request type: {request_type}")
                
        except Exception as e:
            logger.error(f"Error processing ML request: {e}")
            await self._send_error(websocket, str(e))
            
    async def _handle_forecast_request(self, websocket: WebSocket, request_data: Dict[str, Any]):
        """Handle real-time forecasting request"""
        try:
            # Send acknowledgment
            await self._send_response(websocket, {
                'type': 'forecast_started',
                'message': 'Starting real-time forecasting...',
                'timestamp': datetime.now().isoformat()
            })
            
            # Get request parameters
            model_type = request_data.get('model', 'prophet')
            periods = request_data.get('periods', 7)
            historical_data = request_data.get('data', [])
            
            # Convert data to DataFrame
            if historical_data:
                df = pd.DataFrame(historical_data)
            else:
                # Use sample data
                df = self._generate_sample_data()
            
            # Perform forecasting
            if model_type == 'prophet' and self.production_loader.is_prophet_available():
                forecaster = self.production_loader.load_prophet_model()
                if forecaster:
                    predictions = forecaster.predict(periods=periods)
                    await self._send_forecast_results(websocket, predictions, model_type)
                else:
                    await self._send_error(websocket, "Prophet model not available")
                    
            elif model_type == 'arima' and self.production_loader.is_arima_available():
                forecaster = self.production_loader.load_arima_model()
                if forecaster:
                    predictions = forecaster.predict(periods=periods)
                    await self._send_forecast_results(websocket, predictions, model_type)
                else:
                    await self._send_error(websocket, "ARIMA model not available")
                    
            else:
                # Use basic forecasting
                predictions = self._generate_basic_forecast(df, periods)
                await self._send_forecast_results(websocket, predictions, 'basic')
                
        except Exception as e:
            await self._send_error(websocket, f"Forecast error: {str(e)}")
            
    async def _handle_risk_assessment_request(self, websocket: WebSocket, request_data: Dict[str, Any]):
        """Handle real-time risk assessment request"""
        try:
            # Send acknowledgment
            await self._send_response(websocket, {
                'type': 'risk_assessment_started',
                'message': 'Starting real-time risk assessment...',
                'timestamp': datetime.now().isoformat()
            })
            
            # Get supplier data
            supplier_data = request_data.get('supplier_data', {})
            
            # Perform risk assessment
            if self.production_loader.is_risk_model_available():
                risk_assessor = self.production_loader.load_risk_model()
                if risk_assessor:
                    risk_score = risk_assessor.assess_single_supplier(supplier_data)
                    await self._send_risk_results(websocket, risk_score)
                else:
                    await self._send_error(websocket, "Risk assessment model not available")
            else:
                # Use basic risk assessment
                risk_score = self._generate_basic_risk_score(supplier_data)
                await self._send_risk_results(websocket, risk_score)
                
        except Exception as e:
            await self._send_error(websocket, f"Risk assessment error: {str(e)}")
            
    async def _handle_optimization_request(self, websocket: WebSocket, request_data: Dict[str, Any]):
        """Handle real-time optimization request"""
        try:
            # Send acknowledgment
            await self._send_response(websocket, {
                'type': 'optimization_started',
                'message': 'Starting real-time optimization...',
                'timestamp': datetime.now().isoformat()
            })
            
            # Get optimization parameters
            materials = request_data.get('materials', [])
            suppliers = request_data.get('suppliers', [])
            constraints = request_data.get('constraints', {})
            
            # Perform optimization
            optimization_results = self._perform_optimization(materials, suppliers, constraints)
            
            await self._send_response(websocket, {
                'type': 'optimization_complete',
                'results': optimization_results,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            await self._send_error(websocket, f"Optimization error: {str(e)}")
            
    async def _handle_status_request(self, websocket: WebSocket, request_data: Dict[str, Any]):
        """Handle system status request"""
        try:
            status = {
                'type': 'system_status',
                'models': {
                    'prophet': self.production_loader.is_prophet_available(),
                    'arima': self.production_loader.is_arima_available(),
                    'xgboost': self.production_loader.is_xgboost_available(),
                    'lstm': self.production_loader.is_lstm_available(),
                    'risk_model': self.production_loader.is_risk_model_available()
                },
                'performance': self.performance_monitor.get_current_metrics(),
                'connections': len(self.active_connections),
                'timestamp': datetime.now().isoformat()
            }
            
            await self._send_response(websocket, status)
            
        except Exception as e:
            await self._send_error(websocket, f"Status error: {str(e)}")
            
    async def _send_forecast_results(self, websocket: WebSocket, predictions: pd.DataFrame, model_type: str):
        """Send forecast results to WebSocket"""
        try:
            # Convert predictions to JSON-serializable format
            if isinstance(predictions, pd.DataFrame):
                forecast_data = predictions.to_dict('records')
            else:
                forecast_data = predictions
                
            response = {
                'type': 'forecast_complete',
                'model_type': model_type,
                'predictions': forecast_data,
                'timestamp': datetime.now().isoformat()
            }
            
            await self._send_response(websocket, response)
            
        except Exception as e:
            await self._send_error(websocket, f"Error sending forecast results: {str(e)}")
            
    async def _send_risk_results(self, websocket: WebSocket, risk_score: Dict[str, Any]):
        """Send risk assessment results to WebSocket"""
        try:
            response = {
                'type': 'risk_assessment_complete',
                'risk_score': risk_score,
                'timestamp': datetime.now().isoformat()
            }
            
            await self._send_response(websocket, response)
            
        except Exception as e:
            await self._send_error(websocket, f"Error sending risk results: {str(e)}")
            
    async def _send_response(self, websocket: WebSocket, response: Dict[str, Any]):
        """Send JSON response to WebSocket"""
        await websocket.send_text(json.dumps(response))
        
    async def _send_error(self, websocket: WebSocket, error_message: str):
        """Send error response to WebSocket"""
        response = {
            'type': 'error',
            'message': error_message,
            'timestamp': datetime.now().isoformat()
        }
        await websocket.send_text(json.dumps(response))
        
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample data for testing"""
        dates = pd.date_range(start='2025-01-01', periods=30, freq='D')
        demand = np.random.normal(1000, 100, 30)
        
        return pd.DataFrame({
            'date': dates,
            'demand': demand
        })
        
    def _generate_basic_forecast(self, df: pd.DataFrame, periods: int) -> List[Dict[str, Any]]:
        """Generate basic forecast using simple methods"""
        if 'demand' in df.columns:
            recent_avg = df['demand'].tail(7).mean()
            trend = (df['demand'].tail(7).mean() - df['demand'].head(7).mean()) / len(df)
        else:
            recent_avg = 1000
            trend = 0
            
        forecasts = []
        for i in range(periods):
            forecast_value = recent_avg + (trend * i)
            forecasts.append({
                'period': i + 1,
                'forecast': forecast_value,
                'confidence': 0.8
            })
            
        return forecasts
        
    def _generate_basic_risk_score(self, supplier_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic risk score"""
        # Simple risk scoring based on available data
        reliability = supplier_data.get('reliability_score', 0.8)
        lead_time = supplier_data.get('lead_time_days', 14)
        cost_factor = supplier_data.get('cost_factor', 1.0)
        
        # Calculate risk score (lower is better)
        risk_score = (1 - reliability) * 0.5 + (lead_time / 30) * 0.3 + (cost_factor - 1) * 0.2
        
        return {
            'risk_score': min(risk_score, 1.0),
            'risk_level': 'low' if risk_score < 0.3 else 'medium' if risk_score < 0.7 else 'high',
            'factors': {
                'reliability': reliability,
                'lead_time': lead_time,
                'cost_factor': cost_factor
            }
        }
        
    def _perform_optimization(self, materials: List[Dict], suppliers: List[Dict], constraints: Dict) -> Dict[str, Any]:
        """Perform basic optimization"""
        # Simple optimization logic
        optimized_suppliers = []
        
        for material in materials:
            material_suppliers = [s for s in suppliers if material['id'] in s.get('materials', [])]
            if material_suppliers:
                # Select best supplier based on cost and reliability
                best_supplier = min(material_suppliers, key=lambda s: s.get('cost', 1.0) * (1 - s.get('reliability', 0.8)))
                optimized_suppliers.append({
                    'material_id': material['id'],
                    'supplier_id': best_supplier['id'],
                    'cost': best_supplier.get('cost', 1.0),
                    'reliability': best_supplier.get('reliability', 0.8)
                })
                
        return {
            'optimized_suppliers': optimized_suppliers,
            'total_cost': sum(s['cost'] for s in optimized_suppliers),
            'average_reliability': sum(s['reliability'] for s in optimized_suppliers) / len(optimized_suppliers) if optimized_suppliers else 0
        }

# Create global WebSocket manager
websocket_manager = WebSocketManager()

@websocket_router.websocket("/ws/ml-inference")
async def websocket_ml_inference(websocket: WebSocket):
    """WebSocket endpoint for real-time ML inference"""
    await websocket_manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                request_data = json.loads(data)
                await websocket_manager.process_ml_request(websocket, request_data)
                
            except json.JSONDecodeError:
                await websocket_manager._send_error(websocket, "Invalid JSON format")
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
        
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket)

@websocket_router.websocket("/ws/monitoring")
async def websocket_monitoring(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring"""
    await websocket_manager.connect(websocket)
    
    try:
        while True:
            # Send periodic status updates
            await asyncio.sleep(5)  # Update every 5 seconds
            
            status = {
                'type': 'monitoring_update',
                'metrics': websocket_manager.performance_monitor.get_current_metrics(),
                'timestamp': datetime.now().isoformat()
            }
            
            await websocket_manager.send_personal_message(json.dumps(status), websocket)
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
        logger.info("Monitoring WebSocket client disconnected")
        
    except Exception as e:
        logger.error(f"Monitoring WebSocket error: {e}")
        websocket_manager.disconnect(websocket)