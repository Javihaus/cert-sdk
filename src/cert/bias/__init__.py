"""
CERT SDK Bias Detection Module

Two detection modes:
1. Demographic Bias - Standard categories (gender, race, etc.)
2. Custom Policies - Domain-specific rules

Usage:
    from cert.bias import DemographicBiasConfig, BiasSeverity
    from cert.bias.templates import get_template
"""

from .models import (
    # Enums
    BiasSeverity,
    BiasConsensus,

    # Demographic bias
    DemographicCategory,
    DemographicBiasConfig,

    # Custom policies
    PolicyDimension,
    CustomPolicy,

    # Results (for type hints when reading back)
    CategoryResult,
    DimensionResult,
    BiasEvaluationResult,
)

from .templates import (
    get_template,
    list_templates,
    POLICY_TEMPLATES,

    # Individual templates
    FINANCIAL_SERVICES_POLICY,
    HEALTHCARE_EQUITY_POLICY,
    HR_RECRUITMENT_POLICY,
    LEGAL_SERVICES_POLICY,
)

__all__ = [
    # Enums
    "BiasSeverity",
    "BiasConsensus",

    # Configuration
    "DemographicCategory",
    "DemographicBiasConfig",
    "PolicyDimension",
    "CustomPolicy",

    # Results
    "CategoryResult",
    "DimensionResult",
    "BiasEvaluationResult",

    # Templates
    "get_template",
    "list_templates",
    "POLICY_TEMPLATES",
    "FINANCIAL_SERVICES_POLICY",
    "HEALTHCARE_EQUITY_POLICY",
    "HR_RECRUITMENT_POLICY",
    "LEGAL_SERVICES_POLICY",
]
