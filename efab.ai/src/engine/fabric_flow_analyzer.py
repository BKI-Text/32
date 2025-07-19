#!/usr/bin/env python3
"""
Fabric Flow Analyzer for Beverly Knits AI Supply Chain Planner
Tracks and analyzes fabric inventory flow through manufacturing stages
"""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date, timedelta
from collections import defaultdict
from dataclasses import dataclass
import pandas as pd
from enum import Enum

from src.core.domain.entities import (
    FabricInventory, FabricInventoryStatus, MaterialId, Material, MaterialType
)
from src.core.domain.value_objects import Quantity, Money
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)

@dataclass
class FabricFlowMetrics:
    """Metrics for fabric flow analysis"""
    total_quantity: float
    status_distribution: Dict[str, float]
    avg_cycle_time_days: Optional[float]
    bottleneck_stages: List[str]
    quality_yield_rate: float
    allocation_rate: float
    
@dataclass
class FlowStageAnalysis:
    """Analysis of a specific fabric flow stage"""
    stage: FabricInventoryStatus
    total_quantity: float
    average_dwell_time_days: float
    throughput_rate: float
    quality_issues: int
    capacity_utilization: float

class FabricFlowState(str, Enum):
    """Fabric flow state categories for analysis"""
    RAW_MATERIALS = "raw_materials"  # G00
    IN_PRODUCTION = "in_production"  # G02, G04, I01, T01
    FINISHED_GOODS = "finished_goods"  # F01, F02
    ALLOCATED = "allocated"  # P01
    QUARANTINED = "quarantined"  # F08, F09, G09
    SHIPPED = "shipped"  # BH (billed and held)

class FabricFlowAnalyzer:
    """Analyzes fabric inventory flow and identifies optimization opportunities"""
    
    def __init__(self):
        self.fabric_inventories: List[FabricInventory] = []
        self.flow_history: List[Dict[str, Any]] = []
        
        # Define the standard fabric flow paths
        self.standard_flow_paths = {
            "internal_manufacture": [
                FabricInventoryStatus.G00_GREIGE_GOODS,
                FabricInventoryStatus.G02_INTERNAL_MANUFACTURE,
                FabricInventoryStatus.I01_AWAITING_INSPECTION,
                FabricInventoryStatus.F01_FINISHED_INVENTORY,
                FabricInventoryStatus.P01_ALLOCATED
            ],
            "external_manufacture": [
                FabricInventoryStatus.G04_EXTERNAL_MANUFACTURE,
                FabricInventoryStatus.I01_AWAITING_INSPECTION,
                FabricInventoryStatus.F01_FINISHED_INVENTORY,
                FabricInventoryStatus.P01_ALLOCATED
            ],
            "external_purchase": [
                FabricInventoryStatus.F02_EXTERNAL_FINISHED,
                FabricInventoryStatus.P01_ALLOCATED
            ]
        }
        
        # Expected cycle times (in days) for each stage
        self.expected_cycle_times = {
            FabricInventoryStatus.G00_GREIGE_GOODS: 2,
            FabricInventoryStatus.G02_INTERNAL_MANUFACTURE: 5,
            FabricInventoryStatus.G04_EXTERNAL_MANUFACTURE: 7,
            FabricInventoryStatus.I01_AWAITING_INSPECTION: 1,
            FabricInventoryStatus.T01_AWAITING_TEST: 3,
            FabricInventoryStatus.F01_FINISHED_INVENTORY: 0,  # Ready to ship
            FabricInventoryStatus.F02_EXTERNAL_FINISHED: 0,  # Ready to ship
            FabricInventoryStatus.P01_ALLOCATED: 0,  # Allocated
        }
    
    def load_fabric_inventory(self, fabric_inventories: List[FabricInventory]):
        """Load fabric inventory data for analysis"""
        self.fabric_inventories = fabric_inventories
        logger.info(f"Loaded {len(fabric_inventories)} fabric inventory records")
    
    def analyze_fabric_flow(self) -> FabricFlowMetrics:
        """Analyze overall fabric flow metrics"""
        if not self.fabric_inventories:
            logger.warning("No fabric inventory data loaded")
            return FabricFlowMetrics(
                total_quantity=0,
                status_distribution={},
                avg_cycle_time_days=None,
                bottleneck_stages=[],
                quality_yield_rate=0,
                allocation_rate=0
            )
        
        # Calculate status distribution
        status_quantities = defaultdict(float)
        total_quantity = 0
        
        for fabric in self.fabric_inventories:
            qty = float(fabric.quantity.amount)
            status_quantities[fabric.status.value] += qty
            total_quantity += qty
        
        status_distribution = {
            status: (qty / total_quantity * 100) if total_quantity > 0 else 0
            for status, qty in status_quantities.items()
        }
        
        # Calculate quality yield rate
        quality_yield_rate = self._calculate_quality_yield_rate()
        
        # Calculate allocation rate
        allocation_rate = self._calculate_allocation_rate()
        
        # Identify bottlenecks
        bottleneck_stages = self._identify_bottlenecks()
        
        # Calculate average cycle time (simplified)
        avg_cycle_time = self._calculate_average_cycle_time()
        
        return FabricFlowMetrics(
            total_quantity=total_quantity,
            status_distribution=status_distribution,
            avg_cycle_time_days=avg_cycle_time,
            bottleneck_stages=bottleneck_stages,
            quality_yield_rate=quality_yield_rate,
            allocation_rate=allocation_rate
        )
    
    def analyze_stage_performance(self) -> List[FlowStageAnalysis]:
        """Analyze performance of each fabric flow stage"""
        stage_analyses = []
        
        for status in FabricInventoryStatus:
            stage_fabrics = [f for f in self.fabric_inventories if f.status == status]
            
            if not stage_fabrics:
                continue
                
            total_qty = sum(float(f.quantity.amount) for f in stage_fabrics)
            
            # Calculate average dwell time
            avg_dwell_time = self._calculate_stage_dwell_time(status, stage_fabrics)
            
            # Calculate throughput rate (simplified)
            expected_time = self.expected_cycle_times.get(status, 1)
            throughput_rate = total_qty / max(expected_time, 0.1) if expected_time > 0 else total_qty
            
            # Count quality issues
            quality_issues = len([f for f in stage_fabrics if f.is_quarantined()])
            
            # Calculate capacity utilization (simplified)
            capacity_utilization = min(avg_dwell_time / expected_time, 2.0) if expected_time > 0 else 1.0
            
            stage_analyses.append(FlowStageAnalysis(
                stage=status,
                total_quantity=total_qty,
                average_dwell_time_days=avg_dwell_time,
                throughput_rate=throughput_rate,
                quality_issues=quality_issues,
                capacity_utilization=capacity_utilization
            ))
        
        return sorted(stage_analyses, key=lambda x: x.total_quantity, reverse=True)
    
    def get_fabric_flow_state_summary(self) -> Dict[FabricFlowState, float]:
        """Get summary of fabric quantities by flow state category"""
        state_quantities = defaultdict(float)
        
        # Map statuses to flow states
        status_to_state = {
            FabricInventoryStatus.G00_GREIGE_GOODS: FabricFlowState.RAW_MATERIALS,
            FabricInventoryStatus.G02_INTERNAL_MANUFACTURE: FabricFlowState.IN_PRODUCTION,
            FabricInventoryStatus.G04_EXTERNAL_MANUFACTURE: FabricFlowState.IN_PRODUCTION,
            FabricInventoryStatus.I01_AWAITING_INSPECTION: FabricFlowState.IN_PRODUCTION,
            FabricInventoryStatus.T01_AWAITING_TEST: FabricFlowState.IN_PRODUCTION,
            FabricInventoryStatus.F01_FINISHED_INVENTORY: FabricFlowState.FINISHED_GOODS,
            FabricInventoryStatus.F02_EXTERNAL_FINISHED: FabricFlowState.FINISHED_GOODS,
            FabricInventoryStatus.P01_ALLOCATED: FabricFlowState.ALLOCATED,
            FabricInventoryStatus.F08_QUARANTINED_QUALITY: FabricFlowState.QUARANTINED,
            FabricInventoryStatus.F09_SECOND_QUALITY: FabricFlowState.QUARANTINED,
            FabricInventoryStatus.G09_SECOND_QUALITY_GREIGE: FabricFlowState.QUARANTINED,
            FabricInventoryStatus.BH_BILLED_HELD: FabricFlowState.SHIPPED,
        }
        
        for fabric in self.fabric_inventories:
            flow_state = status_to_state.get(fabric.status, FabricFlowState.IN_PRODUCTION)
            state_quantities[flow_state] += float(fabric.quantity.amount)
        
        return dict(state_quantities)
    
    def identify_slow_moving_inventory(self, days_threshold: int = 30) -> List[FabricInventory]:
        """Identify fabric inventory that has been in the same status too long"""
        slow_moving = []
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        
        for fabric in self.fabric_inventories:
            if fabric.last_status_change < cutoff_date:
                # Check if it's stuck in a production stage
                if fabric.is_in_production():
                    slow_moving.append(fabric)
        
        return sorted(slow_moving, key=lambda x: x.last_status_change)
    
    def get_allocation_opportunities(self) -> List[FabricInventory]:
        """Get fabric inventory available for sales order allocation"""
        return [f for f in self.fabric_inventories if f.is_available_for_allocation()]
    
    def get_quality_control_queue(self) -> List[FabricInventory]:
        """Get fabric inventory awaiting quality control processes"""
        qc_statuses = {
            FabricInventoryStatus.I01_AWAITING_INSPECTION,
            FabricInventoryStatus.T01_AWAITING_TEST
        }
        return [f for f in self.fabric_inventories if f.status in qc_statuses]
    
    def _calculate_quality_yield_rate(self) -> float:
        """Calculate overall quality yield rate"""
        total_processed = len([f for f in self.fabric_inventories 
                              if f.status not in {FabricInventoryStatus.G00_GREIGE_GOODS}])
        
        if total_processed == 0:
            return 1.0
        
        quarantined = len([f for f in self.fabric_inventories if f.is_quarantined()])
        return max(0, (total_processed - quarantined) / total_processed)
    
    def _calculate_allocation_rate(self) -> float:
        """Calculate rate of finished goods allocation"""
        finished_goods = len([f for f in self.fabric_inventories 
                             if f.status in {FabricInventoryStatus.F01_FINISHED_INVENTORY,
                                           FabricInventoryStatus.F02_EXTERNAL_FINISHED}])
        
        allocated = len([f for f in self.fabric_inventories 
                        if f.status == FabricInventoryStatus.P01_ALLOCATED])
        
        total_allocatable = finished_goods + allocated
        return allocated / total_allocatable if total_allocatable > 0 else 0
    
    def _identify_bottlenecks(self) -> List[str]:
        """Identify potential bottleneck stages"""
        bottlenecks = []
        stage_analyses = self.analyze_stage_performance()
        
        for analysis in stage_analyses:
            # High capacity utilization indicates potential bottleneck
            if analysis.capacity_utilization > 1.5:
                bottlenecks.append(analysis.stage.value)
            
            # High dwell time compared to expected
            expected_time = self.expected_cycle_times.get(analysis.stage, 1)
            if analysis.average_dwell_time_days > expected_time * 2:
                bottlenecks.append(analysis.stage.value)
        
        return list(set(bottlenecks))
    
    def _calculate_average_cycle_time(self) -> Optional[float]:
        """Calculate average cycle time across all fabrics"""
        cycle_times = []
        
        for fabric in self.fabric_inventories:
            days_in_status = (datetime.now() - fabric.last_status_change).days
            expected_time = self.expected_cycle_times.get(fabric.status, 1)
            
            # Only include if fabric has moved through the system
            if days_in_status > 0:
                cycle_times.append(days_in_status)
        
        return sum(cycle_times) / len(cycle_times) if cycle_times else None
    
    def _calculate_stage_dwell_time(self, status: FabricInventoryStatus, 
                                   stage_fabrics: List[FabricInventory]) -> float:
        """Calculate average dwell time for a specific stage"""
        dwell_times = []
        
        for fabric in stage_fabrics:
            days_in_status = (datetime.now() - fabric.last_status_change).days
            dwell_times.append(max(days_in_status, 0))
        
        return sum(dwell_times) / len(dwell_times) if dwell_times else 0
    
    def generate_flow_report(self) -> Dict[str, Any]:
        """Generate comprehensive fabric flow analysis report"""
        flow_metrics = self.analyze_fabric_flow()
        stage_performance = self.analyze_stage_performance()
        flow_state_summary = self.get_fabric_flow_state_summary()
        slow_moving = self.identify_slow_moving_inventory()
        allocation_opportunities = self.get_allocation_opportunities()
        qc_queue = self.get_quality_control_queue()
        
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_fabric_quantity": flow_metrics.total_quantity,
                "total_records": len(self.fabric_inventories),
                "quality_yield_rate": flow_metrics.quality_yield_rate,
                "allocation_rate": flow_metrics.allocation_rate,
                "avg_cycle_time_days": flow_metrics.avg_cycle_time_days,
                "bottleneck_stages": flow_metrics.bottleneck_stages
            },
            "status_distribution": flow_metrics.status_distribution,
            "flow_state_summary": {state.value: qty for state, qty in flow_state_summary.items()},
            "stage_performance": [
                {
                    "stage": analysis.stage.value,
                    "total_quantity": analysis.total_quantity,
                    "avg_dwell_time_days": analysis.average_dwell_time_days,
                    "throughput_rate": analysis.throughput_rate,
                    "quality_issues": analysis.quality_issues,
                    "capacity_utilization": analysis.capacity_utilization
                }
                for analysis in stage_performance
            ],
            "alerts": {
                "slow_moving_inventory": len(slow_moving),
                "quality_control_queue": len(qc_queue),
                "allocation_opportunities": len(allocation_opportunities),
                "quarantined_items": len([f for f in self.fabric_inventories if f.is_quarantined()])
            },
            "recommendations": self._generate_recommendations(flow_metrics, stage_performance, slow_moving)
        }
    
    def _generate_recommendations(self, flow_metrics: FabricFlowMetrics, 
                                stage_performance: List[FlowStageAnalysis],
                                slow_moving: List[FabricInventory]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Quality yield recommendations
        if flow_metrics.quality_yield_rate < 0.95:
            recommendations.append(
                f"Quality yield rate is {flow_metrics.quality_yield_rate:.1%}. "
                "Consider reviewing quality control processes and supplier performance."
            )
        
        # Allocation rate recommendations
        if flow_metrics.allocation_rate < 0.80:
            recommendations.append(
                f"Allocation rate is {flow_metrics.allocation_rate:.1%}. "
                "Consider improving demand forecasting or sales order processing."
            )
        
        # Bottleneck recommendations
        if flow_metrics.bottleneck_stages:
            recommendations.append(
                f"Bottlenecks identified in stages: {', '.join(flow_metrics.bottleneck_stages)}. "
                "Consider capacity planning or process optimization."
            )
        
        # Slow moving inventory
        if len(slow_moving) > 10:
            recommendations.append(
                f"{len(slow_moving)} items are slow-moving. "
                "Review for potential expediting or alternative uses."
            )
        
        # High-capacity utilization stages
        high_util_stages = [s for s in stage_performance if s.capacity_utilization > 1.5]
        if high_util_stages:
            stage_names = [s.stage.value for s in high_util_stages]
            recommendations.append(
                f"High capacity utilization in stages: {', '.join(stage_names)}. "
                "Consider adding capacity or improving efficiency."
            )
        
        return recommendations