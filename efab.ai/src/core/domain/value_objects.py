from decimal import Decimal
from typing import Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime

class Money(BaseModel):
    amount: Decimal = Field(..., ge=0, description="Monetary amount")
    currency: str = Field(default="USD", description="Currency code")
    
    def __add__(self, other):
        if isinstance(other, Money):
            if self.currency != other.currency:
                raise ValueError(f"Cannot add different currencies: {self.currency} and {other.currency}")
            return Money(amount=self.amount + other.amount, currency=self.currency)
        return Money(amount=self.amount + Decimal(str(other)), currency=self.currency)
    
    def __mul__(self, other):
        return Money(amount=self.amount * Decimal(str(other)), currency=self.currency)
    
    def __str__(self):
        return f"{self.currency} {self.amount:.2f}"

class Quantity(BaseModel):
    amount: Decimal = Field(..., ge=0, description="Quantity amount")
    unit: str = Field(..., description="Unit of measurement")
    
    def __add__(self, other):
        if isinstance(other, Quantity):
            if self.unit != other.unit:
                raise ValueError(f"Cannot add different units: {self.unit} and {other.unit}")
            return Quantity(amount=self.amount + other.amount, unit=self.unit)
        return Quantity(amount=self.amount + Decimal(str(other)), unit=self.unit)
    
    def __mul__(self, other):
        return Quantity(amount=self.amount * Decimal(str(other)), unit=self.unit)
    
    def __str__(self):
        return f"{self.amount} {self.unit}"

class MaterialId(BaseModel):
    value: str = Field(..., description="Material identifier")
    
    def __str__(self):
        return self.value

class SupplierId(BaseModel):
    value: str = Field(..., description="Supplier identifier")
    
    def __str__(self):
        return self.value

class SkuId(BaseModel):
    value: str = Field(..., description="SKU identifier")
    
    def __str__(self):
        return self.value

class LeadTime(BaseModel):
    days: int = Field(..., ge=0, description="Lead time in days")
    
    def __str__(self):
        return f"{self.days} days"