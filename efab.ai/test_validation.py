#!/usr/bin/env python3
"""Test script for the validation system"""

import sys
from decimal import Decimal
from datetime import date, timedelta
from typing import Dict, Any

# Add src to path
sys.path.insert(0, 'src')

from validation import (
    MaterialValidationSchema,
    SupplierValidationSchema,
    InventoryValidationSchema,
    ValidationUtils,
    ValidationSchemaRegistry,
    validate_material_data,
    validate_supplier_data,
    validate_inventory_data
)

def test_material_validation():
    """Test material validation schema"""
    print("🧪 Testing Material Validation...")
    
    # Valid material data
    valid_material = {
        "id": "YARN001",
        "name": "Cotton Yarn",
        "type": "yarn",
        "description": "High-quality cotton yarn for knitting",
        "specifications": {
            "weight": "DK",
            "fiber": "100% Cotton"
        },
        "is_critical": True
    }
    
    # Invalid material data
    invalid_material = {
        "id": "invalid-id",  # Should be alphanumeric
        "name": "A",  # Too short
        "type": "unknown",  # Invalid type
        "description": "A" * 1001,  # Too long
        "specifications": {
            "weight": "A" * 51,  # Key too long
            "fiber": "A" * 201  # Value too long
        },
        "is_critical": "yes"  # Should be boolean
    }
    
    # Test valid data
    try:
        validated_model = MaterialValidationSchema(**valid_material)
        print("✅ Valid material data passed validation")
        
        # Check validation context
        context = validated_model.validation_context
        if context.has_warnings():
            print(f"⚠️  Material validation has {len(context.get_warnings())} warnings")
            for warning in context.get_warnings():
                print(f"   - {warning.field}: {warning.message}")
        
    except Exception as e:
        print(f"❌ Valid material data failed validation: {e}")
    
    # Test invalid data
    try:
        validated_model = MaterialValidationSchema(**invalid_material)
        print("❌ Invalid material data passed validation (should have failed)")
    except Exception as e:
        print(f"✅ Invalid material data correctly failed validation: {e}")
    
    print()

def test_supplier_validation():
    """Test supplier validation schema"""
    print("🧪 Testing Supplier Validation...")
    
    # Valid supplier data
    valid_supplier = {
        "id": "SUP001",
        "name": "Textile Supplier Inc",
        "contact_info": "contact@supplier.com",
        "lead_time_days": 14,
        "reliability_score": 0.95,
        "risk_level": "low",
        "is_active": True
    }
    
    # Invalid supplier data
    invalid_supplier = {
        "id": "invalid-id",  # Should be alphanumeric
        "name": "A",  # Too short
        "contact_info": "123",  # Too short
        "lead_time_days": 0,  # Should be >= 1
        "reliability_score": 1.5,  # Should be <= 1
        "risk_level": "unknown",  # Invalid risk level
        "is_active": "yes"  # Should be boolean
    }
    
    # Test valid data
    try:
        validated_model = SupplierValidationSchema(**valid_supplier)
        print("✅ Valid supplier data passed validation")
        
        # Check validation context
        context = validated_model.validation_context
        if context.has_warnings():
            print(f"⚠️  Supplier validation has {len(context.get_warnings())} warnings")
            for warning in context.get_warnings():
                print(f"   - {warning.field}: {warning.message}")
        
    except Exception as e:
        print(f"❌ Valid supplier data failed validation: {e}")
    
    # Test invalid data
    try:
        validated_model = SupplierValidationSchema(**invalid_supplier)
        print("❌ Invalid supplier data passed validation (should have failed)")
    except Exception as e:
        print(f"✅ Invalid supplier data correctly failed validation: {e}")
    
    print()

def test_inventory_validation():
    """Test inventory validation schema"""
    print("🧪 Testing Inventory Validation...")
    
    # Valid inventory data
    valid_inventory = {
        "material_id": "YARN001",
        "on_hand_qty": Decimal("150.5"),
        "unit": "kg",
        "open_po_qty": Decimal("50.0"),
        "po_expected_date": date.today() + timedelta(days=7),
        "safety_stock": Decimal("25.0")
    }
    
    # Invalid inventory data
    invalid_inventory = {
        "material_id": "invalid-id",  # Should be alphanumeric
        "on_hand_qty": Decimal("-10"),  # Should be >= 0
        "unit": "unknown",  # Invalid unit
        "open_po_qty": Decimal("-5"),  # Should be >= 0
        "po_expected_date": date.today() - timedelta(days=1),  # Should be future
        "safety_stock": Decimal("-5")  # Should be >= 0
    }
    
    # Test valid data
    try:
        validated_model = InventoryValidationSchema(**valid_inventory)
        print("✅ Valid inventory data passed validation")
        
        # Check validation context
        context = validated_model.validation_context
        if context.has_warnings():
            print(f"⚠️  Inventory validation has {len(context.get_warnings())} warnings")
            for warning in context.get_warnings():
                print(f"   - {warning.field}: {warning.message}")
        
    except Exception as e:
        print(f"❌ Valid inventory data failed validation: {e}")
    
    # Test invalid data
    try:
        validated_model = InventoryValidationSchema(**invalid_inventory)
        print("❌ Invalid inventory data passed validation (should have failed)")
    except Exception as e:
        print(f"✅ Invalid inventory data correctly failed validation: {e}")
    
    print()

def test_batch_validation():
    """Test batch validation utilities"""
    print("🧪 Testing Batch Validation...")
    
    # Test data - mix of valid and invalid
    test_materials = [
        {
            "id": "YARN001",
            "name": "Cotton Yarn",
            "type": "yarn",
            "is_critical": True
        },
        {
            "id": "YARN002",
            "name": "Wool Yarn",
            "type": "yarn",
            "is_critical": False
        },
        {
            "id": "invalid-id",  # Invalid
            "name": "A",  # Too short
            "type": "unknown",  # Invalid type
            "is_critical": "yes"  # Should be boolean
        },
        {
            "id": "FABRIC001",
            "name": "Cotton Fabric",
            "type": "fabric",
            "is_critical": True
        }
    ]
    
    # Test batch validation
    results = ValidationUtils.validate_batch_data(
        test_materials, 
        MaterialValidationSchema,
        stop_on_first_error=False
    )
    
    print(f"📊 Batch Validation Results:")
    print(f"   Total items: {results['total_items']}")
    print(f"   Valid items: {results['summary']['valid_count']}")
    print(f"   Invalid items: {results['summary']['invalid_count']}")
    print(f"   Total errors: {results['summary']['error_count']}")
    print(f"   Total warnings: {results['summary']['warning_count']}")
    
    # Show invalid items
    if results['invalid_items']:
        print(f"\n❌ Invalid items:")
        for item in results['invalid_items']:
            print(f"   Item {item['index']}: {len(item['errors'])} errors")
            for error in item['errors']:
                print(f"     - {error['field']}: {error['message']}")
    
    print()

def test_convenience_functions():
    """Test convenience validation functions"""
    print("🧪 Testing Convenience Functions...")
    
    # Test material validation
    material_data = {
        "id": "YARN001",
        "name": "Cotton Yarn",
        "type": "yarn"
    }
    
    context = validate_material_data(material_data)
    if context.has_errors():
        print("❌ Material validation failed")
        for error in context.get_errors():
            print(f"   - {error.field}: {error.message}")
    else:
        print("✅ Material validation passed")
    
    # Test supplier validation
    supplier_data = {
        "id": "SUP001",
        "name": "Textile Supplier Inc",
        "lead_time_days": 14,
        "reliability_score": 0.95,
        "risk_level": "low"
    }
    
    context = validate_supplier_data(supplier_data)
    if context.has_errors():
        print("❌ Supplier validation failed")
        for error in context.get_errors():
            print(f"   - {error.field}: {error.message}")
    else:
        print("✅ Supplier validation passed")
    
    print()

def test_schema_registry():
    """Test validation schema registry"""
    print("🧪 Testing Schema Registry...")
    
    # List available schemas
    schemas = ValidationSchemaRegistry.list_schemas()
    print(f"📋 Available schemas: {schemas}")
    
    # Test using registry
    material_data = {
        "id": "YARN001",
        "name": "Cotton Yarn",
        "type": "yarn"
    }
    
    context = ValidationSchemaRegistry.validate_data("material", material_data)
    if context.has_errors():
        print("❌ Registry validation failed")
        for error in context.get_errors():
            print(f"   - {error.field}: {error.message}")
    else:
        print("✅ Registry validation passed")
    
    print()

def main():
    """Run all validation tests"""
    print("🚀 Beverly Knits AI Supply Chain Planner - Validation System Test")
    print("=" * 70)
    
    try:
        test_material_validation()
        test_supplier_validation()
        test_inventory_validation()
        test_batch_validation()
        test_convenience_functions()
        test_schema_registry()
        
        print("✅ All validation tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())