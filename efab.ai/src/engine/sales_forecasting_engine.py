from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date, timedelta
import logging
import pandas as pd
import numpy as np
from collections import defaultdict
import statistics
from dataclasses import dataclass

from ..core.domain import (
    Material, Supplier, SupplierMaterial, Inventory, BOM, Forecast, 
    MaterialType, ForecastSource, RiskLevel,
    Money, Quantity, MaterialId, SupplierId, SkuId, LeadTime
)

logger = logging.getLogger(__name__)

@dataclass
class SalesBasedForecast:
    """Sales-based forecast with statistical confidence metrics."""
    sku_id: SkuId
    forecast_qty: Quantity
    forecast_date: date
    confidence_score: float
    seasonal_factor: float
    trend_factor: float
    variability_score: float
    source_data_points: int

@dataclass
class SafetyStockCalculation:
    """Safety stock calculation results."""
    material_id: MaterialId
    safety_stock_qty: Quantity
    calculation_method: str
    service_level: float
    demand_variability: float
    lead_time_variability: float

class SalesForecastingEngine:
    """Advanced sales-based forecasting engine with statistical analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default sales forecasting configuration."""
        return {
            'lookback_days': 90,
            'planning_horizon_days': 90,
            'min_sales_history_days': 30,
            'safety_stock_method': 'statistical',
            'service_level': 0.95,
            'aggregation_period': 'weekly',
            'seasonality_detection': True,
            'trend_analysis': True,
            'confidence_threshold': 0.7,
            'enable_sales_forecasting': True,
            'use_style_yarn_bom': True
        }
    
    def generate_sales_based_forecasts(
        self, 
        sales_data: pd.DataFrame,
        style_bom_data: pd.DataFrame,
        planning_date: Optional[date] = None
    ) -> List[SalesBasedForecast]:
        """Generate sales-based forecasts using historical sales data."""
        
        if planning_date is None:
            planning_date = date.today()
            
        self.logger.info(f"Generating sales-based forecasts for {planning_date}")
        
        # Prepare sales data
        processed_sales = self._process_sales_data(sales_data)
        
        # Generate style-level forecasts
        style_forecasts = self._generate_style_forecasts(processed_sales, planning_date)
        
        # Convert to SKU forecasts using BOM
        sku_forecasts = self._convert_to_sku_forecasts(style_forecasts, style_bom_data)
        
        self.logger.info(f"Generated {len(sku_forecasts)} sales-based forecasts")
        return sku_forecasts
    
    def _process_sales_data(self, sales_data: pd.DataFrame) -> pd.DataFrame:
        """Process and clean sales data for forecasting."""
        
        # Standardize column names
        sales_data.columns = sales_data.columns.str.lower().str.replace(' ', '_')
        
        # Convert date columns
        date_columns = ['ship_date', 'order_date', 'date', 'sales_date']
        for col in date_columns:
            if col in sales_data.columns:
                sales_data[col] = pd.to_datetime(sales_data[col], errors='coerce')
                break
        
        # Clean quantity columns
        qty_columns = ['quantity', 'qty', 'order_qty', 'ship_qty']
        for col in qty_columns:
            if col in sales_data.columns:
                sales_data[col] = pd.to_numeric(sales_data[col], errors='coerce')
                sales_data = sales_data[sales_data[col] > 0]  # Remove negative/zero quantities
                break
        
        # Remove rows with invalid data
        sales_data = sales_data.dropna(subset=['ship_date', 'quantity'])
        
        # Filter to lookback period
        cutoff_date = datetime.now() - timedelta(days=self.config['lookback_days'])
        sales_data = sales_data[sales_data['ship_date'] >= cutoff_date]
        
        self.logger.info(f"Processed sales data: {len(sales_data)} records")
        return sales_data
    
    def _generate_style_forecasts(
        self, 
        sales_data: pd.DataFrame, 
        planning_date: date
    ) -> Dict[str, SalesBasedForecast]:
        """Generate forecasts at the style level."""
        
        style_forecasts = {}
        
        # Group by style
        style_groups = sales_data.groupby('style_id') if 'style_id' in sales_data.columns else sales_data.groupby('sku_id')
        
        for style_id, style_sales in style_groups:
            try:
                forecast = self._forecast_single_style(style_id, style_sales, planning_date)
                if forecast:
                    style_forecasts[style_id] = forecast
            except Exception as e:
                self.logger.warning(f"Failed to forecast style {style_id}: {e}")
                
        return style_forecasts
    
    def _forecast_single_style(
        self, 
        style_id: str, 
        style_sales: pd.DataFrame, 
        planning_date: date
    ) -> Optional[SalesBasedForecast]:
        """Generate forecast for a single style."""
        
        # Check minimum data requirements
        if len(style_sales) < 3:
            self.logger.debug(f"Insufficient data for style {style_id}: {len(style_sales)} records")
            return None
        
        # Aggregate sales by period
        aggregated_sales = self._aggregate_sales_by_period(style_sales)
        
        if len(aggregated_sales) < 2:
            return None
        
        # Calculate base forecast
        base_forecast = self._calculate_base_forecast(aggregated_sales)
        
        # Apply seasonality adjustments
        seasonal_factor = self._calculate_seasonal_factor(aggregated_sales, planning_date)
        
        # Apply trend adjustments
        trend_factor = self._calculate_trend_factor(aggregated_sales)
        
        # Calculate confidence metrics
        confidence_score = self._calculate_confidence_score(aggregated_sales)
        variability_score = self._calculate_variability_score(aggregated_sales)
        
        # Adjust forecast
        adjusted_forecast = base_forecast * seasonal_factor * trend_factor
        
        # Ensure minimum forecast
        adjusted_forecast = max(adjusted_forecast, 1.0)
        
        return SalesBasedForecast(
            sku_id=SkuId(value=style_id),
            forecast_qty=Quantity(amount=Decimal(str(adjusted_forecast)), unit="unit"),
            forecast_date=planning_date,
            confidence_score=confidence_score,
            seasonal_factor=seasonal_factor,
            trend_factor=trend_factor,
            variability_score=variability_score,
            source_data_points=len(style_sales)
        )
    
    def _aggregate_sales_by_period(self, sales_data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sales data by the configured period."""
        
        if self.config['aggregation_period'] == 'weekly':
            sales_data['period'] = sales_data['ship_date'].dt.to_period('W')
        elif self.config['aggregation_period'] == 'monthly':
            sales_data['period'] = sales_data['ship_date'].dt.to_period('M')
        else:  # daily
            sales_data['period'] = sales_data['ship_date'].dt.to_period('D')
        
        aggregated = sales_data.groupby('period').agg({
            'quantity': 'sum',
            'ship_date': 'first'
        }).reset_index()
        
        return aggregated.sort_values('ship_date')
    
    def _calculate_base_forecast(self, aggregated_sales: pd.DataFrame) -> float:
        """Calculate base forecast using moving average."""
        
        quantities = aggregated_sales['quantity'].values
        
        # Use exponential smoothing for base forecast
        alpha = 0.3  # Smoothing factor
        forecast = quantities[0]
        
        for qty in quantities[1:]:
            forecast = alpha * qty + (1 - alpha) * forecast
        
        return float(forecast)
    
    def _calculate_seasonal_factor(self, aggregated_sales: pd.DataFrame, planning_date: date) -> float:
        """Calculate seasonal adjustment factor."""
        
        if not self.config['seasonality_detection']:
            return 1.0
        
        # Simple seasonal factor based on historical same-period performance
        current_period = self._get_period_key(planning_date)
        
        # Group by same period type (e.g., same month, same week of year)
        period_averages = {}
        
        for _, row in aggregated_sales.iterrows():
            period_key = self._get_period_key(row['ship_date'].date())
            if period_key not in period_averages:
                period_averages[period_key] = []
            period_averages[period_key].append(row['quantity'])
        
        # Calculate average for current period
        if current_period in period_averages and len(period_averages[current_period]) > 0:
            period_avg = statistics.mean(period_averages[current_period])
            overall_avg = statistics.mean(aggregated_sales['quantity'])
            
            if overall_avg > 0:
                seasonal_factor = period_avg / overall_avg
                # Cap seasonal factors to reasonable range
                return max(0.5, min(2.0, seasonal_factor))
        
        return 1.0
    
    def _get_period_key(self, date_obj: date) -> str:
        """Get period key for seasonality analysis."""
        if self.config['aggregation_period'] == 'weekly':
            return f"week_{date_obj.isocalendar()[1]}"
        elif self.config['aggregation_period'] == 'monthly':
            return f"month_{date_obj.month}"
        else:
            return f"day_{date_obj.weekday()}"
    
    def _calculate_trend_factor(self, aggregated_sales: pd.DataFrame) -> float:
        """Calculate trend adjustment factor."""
        
        if not self.config['trend_analysis'] or len(aggregated_sales) < 3:
            return 1.0
        
        quantities = aggregated_sales['quantity'].values
        
        # Simple linear trend calculation
        x = np.arange(len(quantities))
        try:
            slope, _ = np.polyfit(x, quantities, 1)
            
            # Convert slope to trend factor
            if len(quantities) > 0:
                avg_qty = np.mean(quantities)
                if avg_qty > 0:
                    trend_rate = slope / avg_qty
                    # Cap trend factors to reasonable range
                    trend_factor = 1.0 + max(-0.5, min(0.5, trend_rate))
                    return trend_factor
        except Exception as e:
            self.logger.debug(f"Trend calculation failed: {e}")
        
        return 1.0
    
    def _calculate_confidence_score(self, aggregated_sales: pd.DataFrame) -> float:
        """Calculate confidence score based on data quality and consistency."""
        
        if len(aggregated_sales) < 2:
            return 0.3
        
        quantities = aggregated_sales['quantity'].values
        
        # Factors affecting confidence:
        # 1. Number of data points
        data_points_score = min(1.0, len(quantities) / 10.0)
        
        # 2. Consistency (inverse of coefficient of variation)
        if len(quantities) > 1 and np.mean(quantities) > 0:
            cv = np.std(quantities) / np.mean(quantities)
            consistency_score = max(0.0, 1.0 - cv)
        else:
            consistency_score = 0.5
        
        # 3. Recency (more recent data = higher confidence)
        recency_score = 0.8  # Default good recency for processed data
        
        # Combined confidence score
        confidence = (data_points_score * 0.4 + 
                     consistency_score * 0.4 + 
                     recency_score * 0.2)
        
        return max(0.1, min(1.0, confidence))
    
    def _calculate_variability_score(self, aggregated_sales: pd.DataFrame) -> float:
        """Calculate demand variability score."""
        
        if len(aggregated_sales) < 2:
            return 1.0
        
        quantities = aggregated_sales['quantity'].values
        
        if np.mean(quantities) > 0:
            cv = np.std(quantities) / np.mean(quantities)
            return min(2.0, cv)  # Cap at 2.0 for extreme variability
        
        return 1.0
    
    def _convert_to_sku_forecasts(
        self, 
        style_forecasts: Dict[str, SalesBasedForecast], 
        style_bom_data: pd.DataFrame
    ) -> List[SalesBasedForecast]:
        """Convert style forecasts to SKU forecasts using BOM data."""
        
        sku_forecasts = []
        
        if not self.config['use_style_yarn_bom']:
            # Direct conversion if not using BOM
            return list(style_forecasts.values())
        
        # Process BOM data
        bom_data = style_bom_data.copy()
        bom_data.columns = bom_data.columns.str.lower().str.replace(' ', '_')
        
        for style_id, style_forecast in style_forecasts.items():
            # Find BOM entries for this style
            style_boms = bom_data[bom_data['style_id'] == style_id] if 'style_id' in bom_data.columns else pd.DataFrame()
            
            if len(style_boms) == 0:
                # No BOM found, use style as SKU
                sku_forecasts.append(style_forecast)
                continue
            
            # Create forecasts for each BOM component
            for _, bom_row in style_boms.iterrows():
                try:
                    component_id = bom_row.get('yarn_id', bom_row.get('component_id', bom_row.get('sku_id', '')))
                    percentage = float(bom_row.get('percentage', bom_row.get('qty_per_unit', 1.0)))
                    
                    if component_id and percentage > 0:
                        component_forecast = SalesBasedForecast(
                            sku_id=SkuId(value=str(component_id)),
                            forecast_qty=Quantity(
                                amount=style_forecast.forecast_qty.amount * Decimal(str(percentage)),
                                unit=style_forecast.forecast_qty.unit
                            ),
                            forecast_date=style_forecast.forecast_date,
                            confidence_score=style_forecast.confidence_score * 0.9,  # Slightly lower confidence for derived forecasts
                            seasonal_factor=style_forecast.seasonal_factor,
                            trend_factor=style_forecast.trend_factor,
                            variability_score=style_forecast.variability_score,
                            source_data_points=style_forecast.source_data_points
                        )
                        sku_forecasts.append(component_forecast)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process BOM for style {style_id}: {e}")
        
        return sku_forecasts
    
    def calculate_statistical_safety_stock(
        self, 
        material_id: MaterialId,
        historical_demand: List[float],
        lead_time_days: int,
        service_level: Optional[float] = None
    ) -> SafetyStockCalculation:
        """Calculate statistical safety stock based on demand variability."""
        
        if service_level is None:
            service_level = self.config['service_level']
        
        if len(historical_demand) < 2:
            # Fallback to percentage-based if insufficient data
            avg_demand = sum(historical_demand) / len(historical_demand) if historical_demand else 0
            safety_stock = avg_demand * 0.15  # 15% default
            
            return SafetyStockCalculation(
                material_id=material_id,
                safety_stock_qty=Quantity(amount=Decimal(str(safety_stock)), unit="unit"),
                calculation_method="percentage_fallback",
                service_level=service_level,
                demand_variability=0.15,
                lead_time_variability=0.0
            )
        
        # Calculate demand statistics
        mean_demand = statistics.mean(historical_demand)
        std_demand = statistics.stdev(historical_demand) if len(historical_demand) > 1 else 0
        
        # Z-score for service level (normal distribution approximation)
        z_scores = {
            0.90: 1.28,
            0.95: 1.65,
            0.97: 1.88,
            0.99: 2.33
        }
        
        z_score = z_scores.get(service_level, 1.65)  # Default to 95%
        
        # Safety stock formula: Z * sqrt(lead_time) * demand_std
        lead_time_factor = math.sqrt(lead_time_days / 7)  # Weekly basis
        safety_stock = z_score * lead_time_factor * std_demand
        
        # Ensure positive
        safety_stock = max(0, safety_stock)
        
        # Calculate variability metrics
        demand_variability = std_demand / mean_demand if mean_demand > 0 else 0
        
        return SafetyStockCalculation(
            material_id=material_id,
            safety_stock_qty=Quantity(amount=Decimal(str(safety_stock)), unit="unit"),
            calculation_method="statistical",
            service_level=service_level,
            demand_variability=demand_variability,
            lead_time_variability=0.1  # Assume 10% lead time variability
        )
    
    def convert_sales_forecasts_to_domain_forecasts(
        self, 
        sales_forecasts: List[SalesBasedForecast]
    ) -> List[Forecast]:
        """Convert sales-based forecasts to domain Forecast objects."""
        
        domain_forecasts = []
        
        for sales_forecast in sales_forecasts:
            # Filter by confidence threshold
            if sales_forecast.confidence_score < self.config['confidence_threshold']:
                continue
            
            domain_forecast = Forecast(
                sku_id=sales_forecast.sku_id,
                forecast_qty=sales_forecast.forecast_qty,
                forecast_date=sales_forecast.forecast_date,
                source=ForecastSource.SALES_HISTORY,
                confidence_score=sales_forecast.confidence_score,
                created_at=datetime.now(),
                notes=f"Sales-based forecast (seasonal: {sales_forecast.seasonal_factor:.2f}, trend: {sales_forecast.trend_factor:.2f})"
            )
            domain_forecasts.append(domain_forecast)
        
        self.logger.info(f"Converted {len(domain_forecasts)} sales forecasts to domain objects")
        return domain_forecasts