"""Database Seeding for Beverly Knits AI Supply Chain Planner"""

from datetime import datetime
from ..auth.auth_service import auth_service
from .repositories.user_repository import UserRepository
from .init_db import init_database

def seed_initial_users():
    """Seed initial users for the application"""
    print("🌱 Seeding initial users...")
    
    user_repo = UserRepository()
    
    # Check if admin user already exists
    admin_user = user_repo.get_by_username("admin")
    if not admin_user:
        print("   Creating admin user...")
        admin_user = auth_service.create_user(
            username="admin",
            email="admin@beverlyknits.com",
            password="admin123",
            full_name="System Administrator",
            role="admin"
        )
        # Verify the admin user
        user_repo.verify_user(admin_user.id)
        print("   ✅ Admin user created: admin / admin123")
    else:
        print("   ⚠️  Admin user already exists")
    
    # Check if planner user already exists
    planner_user = user_repo.get_by_username("planner")
    if not planner_user:
        print("   Creating planner user...")
        planner_user = auth_service.create_user(
            username="planner",
            email="planner@beverlyknits.com",
            password="planner123",
            full_name="Supply Chain Planner",
            role="manager"
        )
        # Verify the planner user
        user_repo.verify_user(planner_user.id)
        print("   ✅ Planner user created: planner / planner123")
    else:
        print("   ⚠️  Planner user already exists")
    
    # Check if viewer user already exists
    viewer_user = user_repo.get_by_username("viewer")
    if not viewer_user:
        print("   Creating viewer user...")
        viewer_user = auth_service.create_user(
            username="viewer",
            email="viewer@beverlyknits.com",
            password="viewer123",
            full_name="Data Viewer",
            role="viewer"
        )
        # Verify the viewer user
        user_repo.verify_user(viewer_user.id)
        print("   ✅ Viewer user created: viewer / viewer123")
    else:
        print("   ⚠️  Viewer user already exists")
    
    print("✅ User seeding completed")

def seed_sample_materials():
    """Seed sample materials for testing"""
    print("🌱 Seeding sample materials...")
    
    from .models.material import MaterialModel
    from .repositories.material_repository import MaterialRepository
    
    material_repo = MaterialRepository()
    
    sample_materials = [
        {
            "name": "Cotton Yarn 100% Organic",
            "type": "yarn",
            "description": "High-quality organic cotton yarn for premium garments",
            "specifications": {
                "weight": "Worsted",
                "color": "Natural",
                "fiber_content": "100% Organic Cotton",
                "yardage": "220 yards"
            },
            "is_critical": True
        },
        {
            "name": "Merino Wool Blend",
            "type": "yarn",
            "description": "Soft merino wool blend for luxury items",
            "specifications": {
                "weight": "DK",
                "color": "Charcoal",
                "fiber_content": "80% Merino Wool, 20% Nylon",
                "yardage": "200 yards"
            },
            "is_critical": True
        },
        {
            "name": "Silk Thread Premium",
            "type": "thread",
            "description": "Premium silk thread for finishing work",
            "specifications": {
                "weight": "40wt",
                "color": "Assorted",
                "fiber_content": "100% Silk",
                "length": "1000m"
            },
            "is_critical": False
        },
        {
            "name": "Cotton Fabric Canvas",
            "type": "fabric",
            "description": "Durable cotton canvas for bags and accessories",
            "specifications": {
                "weight": "12oz",
                "width": "60 inches",
                "fiber_content": "100% Cotton",
                "finish": "Pre-washed"
            },
            "is_critical": False
        },
        {
            "name": "Brass Buttons Set",
            "type": "accessory",
            "description": "Antique brass buttons for vintage-style garments",
            "specifications": {
                "material": "Brass",
                "size": "15mm",
                "finish": "Antique",
                "quantity": "50 pieces"
            },
            "is_critical": False
        }
    ]
    
    created_count = 0
    for material_data in sample_materials:
        existing = material_repo.get_by_name(material_data["name"])
        if not existing:
            material = MaterialModel(**material_data)
            material_repo.create(material)
            created_count += 1
            print(f"   ✅ Created material: {material_data['name']}")
        else:
            print(f"   ⚠️  Material already exists: {material_data['name']}")
    
    print(f"✅ Material seeding completed ({created_count} new materials)")

def seed_sample_suppliers():
    """Seed sample suppliers for testing"""
    print("🌱 Seeding sample suppliers...")
    
    from .models.supplier import SupplierModel
    from .repositories.supplier_repository import SupplierRepository
    
    supplier_repo = SupplierRepository()
    
    sample_suppliers = [
        {
            "name": "Premium Yarn Co.",
            "contact_info": "contact@premiumyarn.com | (555) 123-4567",
            "lead_time_days": 14,
            "reliability_score": 0.95,
            "risk_level": "low",
            "is_active": True
        },
        {
            "name": "Global Textile Supply",
            "contact_info": "orders@globaltextile.com | (555) 234-5678",
            "lead_time_days": 21,
            "reliability_score": 0.85,
            "risk_level": "medium",
            "is_active": True
        },
        {
            "name": "Artisan Craft Materials",
            "contact_info": "info@artisancraft.com | (555) 345-6789",
            "lead_time_days": 10,
            "reliability_score": 0.90,
            "risk_level": "low",
            "is_active": True
        },
        {
            "name": "Budget Textile Wholesale",
            "contact_info": "wholesale@budgettextile.com | (555) 456-7890",
            "lead_time_days": 30,
            "reliability_score": 0.70,
            "risk_level": "high",
            "is_active": True
        }
    ]
    
    created_count = 0
    for supplier_data in sample_suppliers:
        existing = supplier_repo.get_by_name(supplier_data["name"])
        if not existing:
            supplier = SupplierModel(**supplier_data)
            supplier_repo.create(supplier)
            created_count += 1
            print(f"   ✅ Created supplier: {supplier_data['name']}")
        else:
            print(f"   ⚠️  Supplier already exists: {supplier_data['name']}")
    
    print(f"✅ Supplier seeding completed ({created_count} new suppliers)")

def seed_database():
    """Seed the database with initial data"""
    print("🚀 Seeding Beverly Knits AI Supply Chain Planner Database...")
    
    # Initialize database first
    if not init_database():
        print("❌ Database initialization failed. Seeding aborted.")
        return False
    
    try:
        # Seed initial users
        seed_initial_users()
        
        # Seed sample materials
        seed_sample_materials()
        
        # Seed sample suppliers
        seed_sample_suppliers()
        
        print("✅ Database seeding completed successfully!")
        print("\n📋 Default Login Credentials:")
        print("   Admin:   admin / admin123")
        print("   Manager: planner / planner123")
        print("   Viewer:  viewer / viewer123")
        
        return True
        
    except Exception as e:
        print(f"❌ Database seeding failed: {e}")
        return False

if __name__ == "__main__":
    seed_database()