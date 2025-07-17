# CSV Column Mapping - Beverly Knits Raw Material Planner

## Issue Summary
The upload interface expects different column names than what the actual Beverly Knits CSV files contain. This document maps the expected columns to the actual columns in the provided CSV files.

## Column Mappings

### 1. Forecasts File
**Expected columns:** `sku_id`, `forecast_qty`, `forecast_date`, `source`

**Actual file:** Not directly provided - needs to be derived from:
- `Sales Activity Report.csv` - Contains sales history
- `eFab_SO_List.csv` - Contains sales orders
- `cfab_Yarn_Demand_By_Style.csv` - Contains yarn demand by style

**Mapping needed:**
- Generate forecasts from sales data
- Map Style_ID → sku_id
- Calculate forecast_qty from historical sales
- Set forecast_date based on planning horizon
- Set source as "sales_history" or "sales_orders"

### 2. BOMs File
**Expected columns:** `sku_id`, `material_id`, `qty_per_unit`, `unit`

**Actual file:** `Style_BOM.csv`
- Columns: `Style_ID`, `Yarn_ID`, `BOM_Percentage`

**Mapping:**
- `Style_ID` → `sku_id`
- `Yarn_ID` → `material_id`
- `BOM_Percentage` → `qty_per_unit` (needs conversion based on style production unit)
- Add `unit` column (default: "lbs" for yarn)

### 3. Inventory File
**Expected columns:** `material_id`, `on_hand_qty`, `unit`, `open_po_qty`, `po_expected_date`

**Actual file:** `Yarn_ID_Current_Inventory.csv`
- Columns: `Yarn_ID`, `Supplier`, `Description`, `Blend`, `Type`, `Color`, `Desc_1`, `Desc_2`, `Desc_3`, `Inventory`, `On_Order`, `Allocated`, `Planning_Ballance`, `Cost_Pound`, `Total_Cost`

**Mapping:**
- `Yarn_ID` → `material_id`
- `Inventory` → `on_hand_qty` (clean: remove commas, handle negatives)
- Add `unit` column (default: "lbs")
- `On_Order` → `open_po_qty`
- Add `po_expected_date` column (needs to be provided or estimated)

### 4. Suppliers File
**Expected columns:** `material_id`, `supplier_id`, `cost_per_unit`, `lead_time_days`, `moq`, `reliability_score`, `ordering_cost`, `holding_cost_rate`

**Actual files needed:**
- `Supplier_ID.csv` - Contains supplier master data
- `Yarn_ID_Current_Inventory.csv` - Contains yarn-supplier relationships and costs

**Mapping:**
- Join `Yarn_ID` with `Supplier` to create material-supplier relationships
- `Yarn_ID` → `material_id`
- `Supplier` → `supplier_id` (from inventory file)
- `Cost_Pound` → `cost_per_unit` (clean: remove $ and commas)
- `Lead_time` → `lead_time_days` (from Supplier_ID.csv)
- `MOQ` → `moq` (from Supplier_ID.csv)
- Add `reliability_score` (default: 0.95 or calculate from history)
- Add `ordering_cost` (default: 100.0 or from configuration)
- Add `holding_cost_rate` (default: 0.25 or from configuration)

## Data Cleaning Requirements

1. **Numeric fields**: Remove $, commas, parentheses for negatives
2. **Negative inventory**: Convert to 0 for on-hand quantities
3. **Missing values**: Use appropriate defaults
4. **Invalid suppliers**: Filter out "Remove" entries
5. **BOM percentages**: Ensure they sum to 1.0 per SKU

## Recommended Solution

Create a data converter script that:
1. Reads the Beverly Knits CSV files
2. Applies the mappings above
3. Performs necessary data cleaning
4. Outputs files in the expected format for upload