#!/usr/bin/env python3
"""
Beverly Knits Fabric Flow Tracking Demo
Demonstrates the fabric inventory status tracking and flow analysis capabilities
"""

import sys
import os
from datetime import datetime, date, timedelta
from decimal import Decimal
import random
from typing import List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.domain.entities import (
    FabricInventory, FabricInventoryStatus, MaterialId, MaterialType
)
from src.core.domain.value_objects import Quantity
from src.engine.fabric_flow_analyzer import FabricFlowAnalyzer
import json

def generate_sample_fabric_inventory() -> List[FabricInventory]:
    """Generate sample fabric inventory data for demonstration"""
    fabric_inventories = []
    
    # Sample fabric materials
    fabrics = [
        ("COTTON_JERSEY_001", "Cotton Jersey - Navy"),
        ("COTTON_JERSEY_002", "Cotton Jersey - White"), 
        ("POLY_BLEND_001", "Polyester Blend - Black"),
        ("SILK_CREPE_001", "Silk Crepe - Ivory"),
        ("DENIM_001", "Cotton Denim - Blue"),
        ("LYCRA_BLEND_001", "Lycra Blend - Gray"),
        ("WOOL_BLEND_001", "Wool Blend - Charcoal"),
        ("LINEN_001", "Linen - Natural")
    ]
    
    # Generate inventory records across different statuses
    lot_counter = 1000
    
    for fabric_id, fabric_name in fabrics:
        material_id = MaterialId(value=fabric_id)
        
        # Create multiple lots in different stages for each fabric type
        statuses_to_create = [
            (FabricInventoryStatus.G00_GREIGE_GOODS, random.uniform(500, 1500)),
            (FabricInventoryStatus.G02_INTERNAL_MANUFACTURE, random.uniform(300, 800)),
            (FabricInventoryStatus.G04_EXTERNAL_MANUFACTURE, random.uniform(200, 600)),
            (FabricInventoryStatus.I01_AWAITING_INSPECTION, random.uniform(400, 900)),
            (FabricInventoryStatus.F01_FINISHED_INVENTORY, random.uniform(1000, 2500)),
            (FabricInventoryStatus.F02_EXTERNAL_FINISHED, random.uniform(500, 1200)),
            (FabricInventoryStatus.P01_ALLOCATED, random.uniform(200, 800)),
            (FabricInventoryStatus.T01_AWAITING_TEST, random.uniform(100, 400)),
        ]
        
        # Add some quality control and quarantine items occasionally
        if random.random() < 0.3:  # 30% chance of quarantined items
            statuses_to_create.extend([
                (FabricInventoryStatus.F08_QUARANTINED_QUALITY, random.uniform(50, 200)),
                (FabricInventoryStatus.F09_SECOND_QUALITY, random.uniform(100, 300)),
            ])
        
        if random.random() < 0.2:  # 20% chance of second quality greige
            statuses_to_create.append(
                (FabricInventoryStatus.G09_SECOND_QUALITY_GREIGE, random.uniform(100, 400))
            )
        
        if random.random() < 0.1:  # 10% chance of billed and held
            statuses_to_create.append(
                (FabricInventoryStatus.BH_BILLED_HELD, random.uniform(200, 500))
            )
        
        for status, quantity in statuses_to_create:
            lot_counter += 1
            
            # Create realistic timestamps based on status
            days_ago = random.randint(1, 45) if status != FabricInventoryStatus.G00_GREIGE_GOODS else random.randint(1, 10)
            last_change = datetime.now() - timedelta(days=days_ago)
            
            # Determine if allocated
            allocated_to = None
            if status == FabricInventoryStatus.P01_ALLOCATED:
                allocated_to = f"SO_{random.randint(10000, 99999)}"
            
            # Quality grade
            quality_grade = "First"
            if status in [FabricInventoryStatus.F09_SECOND_QUALITY, FabricInventoryStatus.G09_SECOND_QUALITY_GREIGE]:
                quality_grade = "Second"
            elif status == FabricInventoryStatus.F08_QUARANTINED_QUALITY:
                quality_grade = "Questionable"
            
            # Quarantine reason for quarantined items
            quarantine_reason = None
            if status == FabricInventoryStatus.F08_QUARANTINED_QUALITY:
                quarantine_reason = random.choice([
                    "Color variation detected", 
                    "Texture inconsistency", 
                    "Pending additional testing"
                ])
            elif status in [FabricInventoryStatus.F09_SECOND_QUALITY, FabricInventoryStatus.G09_SECOND_QUALITY_GREIGE]:
                quarantine_reason = random.choice([
                    "Minor defects identified",
                    "Below quality standards", 
                    "Customer rejection"
                ])
            
            # Test results for testing/inspection stages
            test_results = {}
            if status in [FabricInventoryStatus.T01_AWAITING_TEST, FabricInventoryStatus.I01_AWAITING_INSPECTION]:
                test_results = {
                    "color_fastness": random.choice(["PASS", "PENDING", "FAIL"]),
                    "tensile_strength": f"{random.uniform(80, 120):.1f} N/cm",
                    "shrinkage_test": random.choice(["PASS", "PENDING"]),
                    "inspector": random.choice(["QC_001", "QC_002", "QC_003"])
                }
            
            fabric_inventory = FabricInventory(
                material_id=material_id,
                status=status,
                quantity=Quantity(amount=Decimal(str(round(quantity, 2))), unit="yards"),
                location=random.choice(["Warehouse_A", "Production_Floor", "QC_Lab", "Finishing_Dept"]),
                lot_number=f"LOT_{lot_counter}",
                quality_grade=quality_grade,
                allocated_to=allocated_to,
                test_results=test_results,
                quarantine_reason=quarantine_reason,
                expected_release_date=date.today() + timedelta(days=random.randint(1, 14)) if status in [
                    FabricInventoryStatus.T01_AWAITING_TEST,
                    FabricInventoryStatus.I01_AWAITING_INSPECTION
                ] else None,
                last_status_change=last_change,
                created_at=last_change - timedelta(days=random.randint(5, 30))
            )
            
            fabric_inventories.append(fabric_inventory)
    
    return fabric_inventories

def main():
    """Main demonstration function"""
    print("🧵 Beverly Knits AI - Fabric Flow Tracking Demo")
    print("=" * 60)
    
    # Generate sample fabric inventory data
    print("\n📊 Generating sample fabric inventory data...")
    fabric_inventories = generate_sample_fabric_inventory()
    print(f"Generated {len(fabric_inventories)} fabric inventory records")
    
    # Initialize fabric flow analyzer
    print("\n🔍 Initializing Fabric Flow Analyzer...")
    analyzer = FabricFlowAnalyzer()
    analyzer.load_fabric_inventory(fabric_inventories)
    
    # Analyze fabric flow
    print("\n📈 Analyzing fabric flow metrics...")
    flow_metrics = analyzer.analyze_fabric_flow()
    
    print(f"\n🎯 FABRIC FLOW SUMMARY:")
    print(f"   Total Fabric Quantity: {flow_metrics.total_quantity:,.1f} yards")
    print(f"   Quality Yield Rate: {flow_metrics.quality_yield_rate:.1%}")
    print(f"   Allocation Rate: {flow_metrics.allocation_rate:.1%}")
    if flow_metrics.avg_cycle_time_days:
        print(f"   Average Cycle Time: {flow_metrics.avg_cycle_time_days:.1f} days")
    
    # Display status distribution
    print(f"\n📊 STATUS DISTRIBUTION:")
    for status, percentage in flow_metrics.status_distribution.items():
        print(f"   {status}: {percentage:.1f}%")
    
    # Identify bottlenecks
    if flow_metrics.bottleneck_stages:
        print(f"\n⚠️  BOTTLENECKS IDENTIFIED:")
        for bottleneck in flow_metrics.bottleneck_stages:
            print(f"   • {bottleneck}")
    
    # Analyze stage performance
    print(f"\n🏭 STAGE PERFORMANCE ANALYSIS:")
    stage_performance = analyzer.analyze_stage_performance()
    for analysis in stage_performance[:5]:  # Show top 5 stages by quantity
        print(f"   {analysis.stage.value}:")
        print(f"     • Quantity: {analysis.total_quantity:,.1f} yards")
        print(f"     • Avg Dwell Time: {analysis.average_dwell_time_days:.1f} days")
        print(f"     • Quality Issues: {analysis.quality_issues}")
        print(f"     • Capacity Utilization: {analysis.capacity_utilization:.1f}")
    
    # Flow state summary
    print(f"\n🌊 FLOW STATE SUMMARY:")
    flow_state_summary = analyzer.get_fabric_flow_state_summary()
    for state, quantity in flow_state_summary.items():
        print(f"   {state.replace('_', ' ').title()}: {quantity:,.1f} yards")
    
    # Identify slow moving inventory
    slow_moving = analyzer.identify_slow_moving_inventory(days_threshold=30)
    print(f"\n⏰ SLOW MOVING INVENTORY: {len(slow_moving)} items")
    if slow_moving:
        print("   Top 3 slowest moving:")
        for fabric in slow_moving[:3]:
            days_stuck = (datetime.now() - fabric.last_status_change).days
            print(f"     • {fabric.material_id.value} - {fabric.status.value} ({days_stuck} days)")
    
    # Allocation opportunities
    allocation_opps = analyzer.get_allocation_opportunities()
    print(f"\n🎯 ALLOCATION OPPORTUNITIES: {len(allocation_opps)} items available")
    total_allocatable = sum(float(f.quantity.amount) for f in allocation_opps)
    print(f"   Total Available for Allocation: {total_allocatable:,.1f} yards")
    
    # Quality control queue
    qc_queue = analyzer.get_quality_control_queue()
    print(f"\n🔬 QUALITY CONTROL QUEUE: {len(qc_queue)} items awaiting QC")
    
    # Generate comprehensive report
    print(f"\n📋 Generating comprehensive fabric flow report...")
    report = analyzer.generate_flow_report()
    
    # Save report to file
    report_filename = f"fabric_flow_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"📄 Report saved to: {report_filename}")
    
    # Display recommendations
    if report["recommendations"]:
        print(f"\n💡 RECOMMENDATIONS:")
        for i, recommendation in enumerate(report["recommendations"], 1):
            print(f"   {i}. {recommendation}")
    
    # Summary of key insights
    print(f"\n🔍 KEY INSIGHTS:")
    print(f"   • {len(fabric_inventories)} fabric lots are being tracked across {len(set(f.status for f in fabric_inventories))} different stages")
    print(f"   • Quality yield rate of {flow_metrics.quality_yield_rate:.1%} indicates {'good' if flow_metrics.quality_yield_rate > 0.95 else 'attention needed for'} quality control")
    print(f"   • Allocation rate of {flow_metrics.allocation_rate:.1%} shows {'efficient' if flow_metrics.allocation_rate > 0.8 else 'room for improvement in'} inventory management")
    print(f"   • {len(slow_moving)} items are slow-moving and may need attention")
    
    print(f"\n✅ Fabric Flow Analysis Complete!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)