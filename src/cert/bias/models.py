"""
Bias detection data models.

These models define the configuration sent to the backend.
The backend performs actual bias evaluation using NLI.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum


class BiasSeverity(str, Enum):
    """Severity levels for bias violations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class BiasConsensus(str, Enum):
    """
    Consensus level for demographic bias categories.

    Based on cross-framework analysis:
    - NIST AI RMF
    - EU AI Act (Article 9)
    - BBQ, CrowS-Pairs, BOLD benchmarks
    """
    UNIVERSAL = "universal"  # All frameworks agree (gender, race, religion)
    STRONG = "strong"        # Most frameworks agree (age, disability, LGBTQ+, socioeconomic)
    MODERATE = "moderate"    # Some frameworks include (nationality)
    EMERGING = "emerging"    # Gaining adoption (political, appearance)


# =============================================================================
# Demographic Bias Configuration
# =============================================================================

@dataclass
class DemographicCategory:
    """
    A single demographic bias category.

    Attributes:
        id: Unique identifier (e.g., "gender", "racial")
        name: Human-readable name
        description: What this category detects
        enabled: Whether to check this category
        threshold: Score threshold for flagging (0-1, higher = stricter)
        severity: How severe violations of this category are
        consensus: Level of cross-framework agreement
    """
    id: str
    name: str
    description: str
    enabled: bool
    threshold: float
    severity: BiasSeverity
    consensus: BiasConsensus

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "threshold": self.threshold,
            "severity": self.severity.value,
            "consensus": self.consensus.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DemographicCategory":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            enabled=data["enabled"],
            threshold=data["threshold"],
            severity=BiasSeverity(data["severity"]),
            consensus=BiasConsensus(data["consensus"]),
        )


@dataclass
class DemographicBiasConfig:
    """
    Configuration for demographic bias detection.

    Usage:
        # Default configuration (standard categories enabled)
        config = DemographicBiasConfig()

        # Customize
        config.enable("political")
        config.set_threshold("gender", 0.7)
        config.set_severity("age", BiasSeverity.HIGH)

        # Use with client
        client = CertClient(..., demographic_bias=config)

    Default enabled categories (UNIVERSAL + STRONG consensus):
        - gender, racial, religious (UNIVERSAL)
        - age, disability, lgbtq, socioeconomic (STRONG)

    Default disabled categories (MODERATE + EMERGING):
        - nationality (MODERATE)
        - political, appearance (EMERGING)
    """
    categories: List[DemographicCategory] = field(default_factory=list)

    def __post_init__(self):
        if not self.categories:
            self.categories = self._default_categories()

    def _default_categories(self) -> List[DemographicCategory]:
        """Create default category configuration."""
        return [
            # UNIVERSAL consensus - enabled by default
            DemographicCategory(
                id="gender",
                name="Gender Bias",
                description="Stereotypes or discrimination based on gender identity",
                enabled=True,
                threshold=0.5,
                severity=BiasSeverity.HIGH,
                consensus=BiasConsensus.UNIVERSAL,
            ),
            DemographicCategory(
                id="racial",
                name="Racial Bias",
                description="Discrimination or stereotyping based on race or ethnicity",
                enabled=True,
                threshold=0.5,
                severity=BiasSeverity.CRITICAL,
                consensus=BiasConsensus.UNIVERSAL,
            ),
            DemographicCategory(
                id="religious",
                name="Religious Bias",
                description="Discrimination based on religious beliefs or practices",
                enabled=True,
                threshold=0.5,
                severity=BiasSeverity.HIGH,
                consensus=BiasConsensus.UNIVERSAL,
            ),

            # STRONG consensus - enabled by default
            DemographicCategory(
                id="age",
                name="Age Bias",
                description="Stereotypes against age groups (elderly, youth)",
                enabled=True,
                threshold=0.5,
                severity=BiasSeverity.MEDIUM,
                consensus=BiasConsensus.STRONG,
            ),
            DemographicCategory(
                id="disability",
                name="Disability Bias",
                description="Discrimination against physical, cognitive, or mental disabilities",
                enabled=True,
                threshold=0.5,
                severity=BiasSeverity.HIGH,
                consensus=BiasConsensus.STRONG,
            ),
            DemographicCategory(
                id="lgbtq",
                name="LGBTQ+ Bias",
                description="Discrimination based on sexual orientation or gender identity",
                enabled=True,
                threshold=0.5,
                severity=BiasSeverity.HIGH,
                consensus=BiasConsensus.STRONG,
            ),
            DemographicCategory(
                id="socioeconomic",
                name="Socioeconomic Bias",
                description="Stereotypes based on income, education, or occupation",
                enabled=True,
                threshold=0.5,
                severity=BiasSeverity.MEDIUM,
                consensus=BiasConsensus.STRONG,
            ),

            # MODERATE consensus - disabled by default
            DemographicCategory(
                id="nationality",
                name="Nationality Bias",
                description="Discrimination based on country of origin or region",
                enabled=False,
                threshold=0.5,
                severity=BiasSeverity.MEDIUM,
                consensus=BiasConsensus.MODERATE,
            ),

            # EMERGING consensus - disabled by default
            DemographicCategory(
                id="political",
                name="Political Bias",
                description="Partisan bias in political or ideological content",
                enabled=False,
                threshold=0.5,
                severity=BiasSeverity.MEDIUM,
                consensus=BiasConsensus.EMERGING,
            ),
            DemographicCategory(
                id="appearance",
                name="Physical Appearance Bias",
                description="Discrimination based on physical characteristics",
                enabled=False,
                threshold=0.5,
                severity=BiasSeverity.LOW,
                consensus=BiasConsensus.EMERGING,
            ),
        ]

    def _get_category(self, category_id: str) -> Optional[DemographicCategory]:
        """Get category by ID, or None if not found."""
        for cat in self.categories:
            if cat.id == category_id:
                return cat
        return None

    def enable(self, category_id: str) -> "DemographicBiasConfig":
        """Enable a bias category."""
        cat = self._get_category(category_id)
        if cat:
            cat.enabled = True
        return self

    def disable(self, category_id: str) -> "DemographicBiasConfig":
        """Disable a bias category."""
        cat = self._get_category(category_id)
        if cat:
            cat.enabled = False
        return self

    def set_threshold(self, category_id: str, threshold: float) -> "DemographicBiasConfig":
        """
        Set detection threshold for a category.

        Args:
            category_id: Category to configure
            threshold: Value between 0 and 1. Higher = stricter (more likely to flag)
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
        cat = self._get_category(category_id)
        if cat:
            cat.threshold = threshold
        return self

    def set_severity(self, category_id: str, severity: BiasSeverity) -> "DemographicBiasConfig":
        """Set severity level for a category."""
        cat = self._get_category(category_id)
        if cat:
            cat.severity = severity
        return self

    def enable_all(self) -> "DemographicBiasConfig":
        """Enable all bias categories."""
        for cat in self.categories:
            cat.enabled = True
        return self

    def disable_all(self) -> "DemographicBiasConfig":
        """Disable all bias categories."""
        for cat in self.categories:
            cat.enabled = False
        return self

    def enable_standard(self) -> "DemographicBiasConfig":
        """Enable only UNIVERSAL and STRONG consensus categories (default behavior)."""
        for cat in self.categories:
            cat.enabled = cat.consensus in [BiasConsensus.UNIVERSAL, BiasConsensus.STRONG]
        return self

    def enable_strict(self) -> "DemographicBiasConfig":
        """Enable all categories and set stricter thresholds."""
        for cat in self.categories:
            cat.enabled = True
            cat.threshold = 0.3  # Lower threshold = more sensitive
        return self

    def get_enabled_categories(self) -> List[DemographicCategory]:
        """Return list of enabled categories."""
        return [cat for cat in self.categories if cat.enabled]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for transmission to backend."""
        return {
            "categories": [cat.to_dict() for cat in self.categories if cat.enabled]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DemographicBiasConfig":
        """Deserialize from dict."""
        categories = [
            DemographicCategory.from_dict(c) for c in data.get("categories", [])
        ]
        return cls(categories=categories)


# =============================================================================
# Custom Policy Configuration
# =============================================================================

@dataclass
class PolicyDimension:
    """
    A dimension within a custom policy.

    Each dimension defines specific fairness requirements that will be
    checked using NLI against the policy statements.

    Attributes:
        name: Unique identifier for this dimension
        display_name: Human-readable name
        statements: Policy statements to check against (premise for NLI)
        severity: How severe violations are
        threshold: Contradiction threshold for flagging (0-1)
        description: What this dimension covers
        task_types: Which task types this dimension applies to ("*" = all)
    """
    name: str
    display_name: str
    statements: List[str]
    severity: BiasSeverity = BiasSeverity.MEDIUM
    threshold: float = 0.5
    description: Optional[str] = None
    task_types: List[str] = field(default_factory=lambda: ["*"])

    def __post_init__(self):
        if not self.statements:
            raise ValueError(f"Dimension '{self.name}' must have at least one statement")
        if not 0 <= self.threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {self.threshold}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "severity": self.severity.value,
            "threshold": self.threshold,
            "statements": self.statements,
            "task_types": self.task_types,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyDimension":
        return cls(
            name=data["name"],
            display_name=data["display_name"],
            statements=data["statements"],
            severity=BiasSeverity(data.get("severity", "medium")),
            threshold=data.get("threshold", 0.5),
            description=data.get("description"),
            task_types=data.get("task_types", ["*"]),
        )


@dataclass
class CustomPolicy:
    """
    Custom bias detection policy for domain-specific rules.

    Custom policies use NLI to check if LLM outputs contradict
    policy statements. Unlike demographic bias (which checks for
    entailment with stereotypes), custom policies check for
    contradiction with fairness requirements.

    Example:
        fair_lending = CustomPolicy(
            name="fair_lending",
            display_name="Fair Lending Policy",
            domain="financial",
            dimensions=[
                PolicyDimension(
                    name="credit_decisions",
                    display_name="Credit Decision Fairness",
                    statements=[
                        "Credit decisions must be based on financial metrics only.",
                        "Demographics must not influence loan recommendations.",
                    ],
                    severity=BiasSeverity.CRITICAL,
                    task_types=["credit_decision", "loan_recommendation"]
                )
            ]
        )
    """
    name: str
    display_name: str
    domain: Literal["financial", "healthcare", "hr", "legal", "general", "custom"]
    dimensions: List[PolicyDimension]
    description: Optional[str] = None

    def __post_init__(self):
        if not self.dimensions:
            raise ValueError(f"Policy '{self.name}' must have at least one dimension")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "domain": self.domain,
            "description": self.description,
            "dimensions": [d.to_dict() for d in self.dimensions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomPolicy":
        dimensions = [
            PolicyDimension.from_dict(d) for d in data.get("dimensions", [])
        ]
        return cls(
            name=data["name"],
            display_name=data["display_name"],
            domain=data["domain"],
            dimensions=dimensions,
            description=data.get("description"),
        )


# =============================================================================
# Result Types (for reading back from API)
# =============================================================================

@dataclass
class CategoryResult:
    """Result for a single demographic bias category."""
    category_id: str
    category_name: str
    score: float
    detected: bool
    severity: str
    threshold_used: float


@dataclass
class DimensionResult:
    """Result for a single custom policy dimension."""
    dimension_name: str
    faithfulness_score: float
    compliant: bool
    severity: str
    threshold_used: float
    violated_statements: Optional[List[str]] = None
    explanation: Optional[str] = None


@dataclass
class BiasEvaluationResult:
    """
    Complete bias evaluation result returned by the backend.

    Use this when parsing bias_results from the API.
    """
    # Demographic bias
    demographic_bias_detected: bool
    demographic_highest_severity: Optional[str]
    demographic_results: List[CategoryResult]

    # Custom policy
    custom_policy_compliant: Optional[bool]
    custom_policy_name: Optional[str]
    custom_policy_results: List[DimensionResult]

    # Summary
    categories_evaluated: int
    categories_flagged: int

    @property
    def overall_clean(self) -> bool:
        """True if no bias detected and all policies compliant."""
        demographic_clean = not self.demographic_bias_detected
        policy_clean = self.custom_policy_compliant is None or self.custom_policy_compliant
        return demographic_clean and policy_clean

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BiasEvaluationResult":
        """Parse from API response."""
        demographic = data.get("demographic", {})
        custom = data.get("custom_policy", {})

        demographic_results = [
            CategoryResult(
                category_id=c["category_id"],
                category_name=c["category_name"],
                score=c["score"],
                detected=c["detected"],
                severity=c["severity"],
                threshold_used=c.get("threshold_used", 0.5),
            )
            for c in demographic.get("categories", [])
        ]

        custom_results = [
            DimensionResult(
                dimension_name=d["dimension_name"],
                faithfulness_score=d["score"],
                compliant=d["compliant"],
                severity=d["severity"],
                threshold_used=d.get("threshold_used", 0.5),
                violated_statements=d.get("violated_statements"),
            )
            for d in custom.get("dimensions", [])
        ]

        return cls(
            demographic_bias_detected=demographic.get("detected", False),
            demographic_highest_severity=demographic.get("highest_severity"),
            demographic_results=demographic_results,
            custom_policy_compliant=custom.get("compliant"),
            custom_policy_name=custom.get("policy_name"),
            custom_policy_results=custom_results,
            categories_evaluated=len(demographic_results),
            categories_flagged=sum(1 for r in demographic_results if r.detected),
        )
