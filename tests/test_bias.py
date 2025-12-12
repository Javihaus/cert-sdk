"""Tests for bias detection module."""

import pytest
from cert.bias.models import (
    DemographicBiasConfig,
    DemographicCategory,
    CustomPolicy,
    PolicyDimension,
    BiasSeverity,
    BiasConsensus,
)
from cert.bias.templates import get_template, list_templates, POLICY_TEMPLATES


class TestDemographicBiasConfig:
    """Tests for DemographicBiasConfig."""

    def test_default_categories_created(self):
        """Default config should have all standard categories."""
        config = DemographicBiasConfig()
        assert len(config.categories) == 10

        # Check some expected categories exist
        ids = [c.id for c in config.categories]
        assert "gender" in ids
        assert "racial" in ids
        assert "political" in ids

    def test_default_enabled_categories(self):
        """UNIVERSAL and STRONG consensus should be enabled by default."""
        config = DemographicBiasConfig()
        enabled = config.get_enabled_categories()

        # Should have 7 enabled by default
        enabled_ids = {c.id for c in enabled}
        assert "gender" in enabled_ids
        assert "racial" in enabled_ids
        assert "political" not in enabled_ids  # EMERGING - disabled
        assert "nationality" not in enabled_ids  # MODERATE - disabled

    def test_enable_category(self):
        """Should enable a disabled category."""
        config = DemographicBiasConfig()
        config.enable("political")

        cat = config._get_category("political")
        assert cat.enabled is True

    def test_disable_category(self):
        """Should disable an enabled category."""
        config = DemographicBiasConfig()
        config.disable("gender")

        cat = config._get_category("gender")
        assert cat.enabled is False

    def test_set_threshold(self):
        """Should set category threshold."""
        config = DemographicBiasConfig()
        config.set_threshold("gender", 0.7)

        cat = config._get_category("gender")
        assert cat.threshold == 0.7

    def test_set_threshold_validation(self):
        """Should reject invalid thresholds."""
        config = DemographicBiasConfig()

        with pytest.raises(ValueError):
            config.set_threshold("gender", 1.5)

        with pytest.raises(ValueError):
            config.set_threshold("gender", -0.1)

    def test_set_severity(self):
        """Should set category severity."""
        config = DemographicBiasConfig()
        config.set_severity("age", BiasSeverity.CRITICAL)

        cat = config._get_category("age")
        assert cat.severity == BiasSeverity.CRITICAL

    def test_enable_all(self):
        """Should enable all categories."""
        config = DemographicBiasConfig()
        config.enable_all()

        for cat in config.categories:
            assert cat.enabled is True

    def test_disable_all(self):
        """Should disable all categories."""
        config = DemographicBiasConfig()
        config.disable_all()

        for cat in config.categories:
            assert cat.enabled is False

    def test_enable_standard(self):
        """Should enable only UNIVERSAL and STRONG categories."""
        config = DemographicBiasConfig()
        config.enable_all()  # First enable all
        config.enable_standard()  # Then apply standard

        for cat in config.categories:
            expected = cat.consensus in [BiasConsensus.UNIVERSAL, BiasConsensus.STRONG]
            assert cat.enabled == expected, f"{cat.id}: expected {expected}, got {cat.enabled}"

    def test_fluent_api(self):
        """Fluent API should return self."""
        config = DemographicBiasConfig()
        result = config.enable("political").set_threshold("gender", 0.7).disable("age")

        assert result is config

    def test_to_dict_only_enabled(self):
        """to_dict should only include enabled categories."""
        config = DemographicBiasConfig()
        config.disable_all()
        config.enable("gender")

        data = config.to_dict()
        assert len(data["categories"]) == 1
        assert data["categories"][0]["id"] == "gender"

    def test_from_dict(self):
        """Should deserialize from dict."""
        original = DemographicBiasConfig()
        original.enable("political").set_threshold("gender", 0.8)

        data = original.to_dict()
        restored = DemographicBiasConfig.from_dict(data)

        # Check restored has same enabled categories
        original_enabled = {c.id for c in original.get_enabled_categories()}
        restored_enabled = {c.id for c in restored.get_enabled_categories()}
        assert original_enabled == restored_enabled


class TestCustomPolicy:
    """Tests for CustomPolicy."""

    def test_create_policy(self):
        """Should create a valid policy."""
        policy = CustomPolicy(
            name="test_policy",
            display_name="Test Policy",
            domain="general",
            dimensions=[
                PolicyDimension(
                    name="test_dim",
                    display_name="Test Dimension",
                    statements=["Test statement"],
                )
            ],
        )

        assert policy.name == "test_policy"
        assert len(policy.dimensions) == 1

    def test_policy_requires_dimensions(self):
        """Should require at least one dimension."""
        with pytest.raises(ValueError):
            CustomPolicy(
                name="empty",
                display_name="Empty",
                domain="general",
                dimensions=[],
            )

    def test_dimension_requires_statements(self):
        """Dimensions should require at least one statement."""
        with pytest.raises(ValueError):
            PolicyDimension(
                name="empty",
                display_name="Empty",
                statements=[],
            )

    def test_dimension_threshold_validation(self):
        """Dimension threshold should be validated."""
        with pytest.raises(ValueError):
            PolicyDimension(
                name="test",
                display_name="Test",
                statements=["Test"],
                threshold=2.0,
            )

    def test_to_dict(self):
        """Should serialize to dict."""
        policy = CustomPolicy(
            name="test",
            display_name="Test",
            domain="financial",
            dimensions=[
                PolicyDimension(
                    name="dim1",
                    display_name="Dim 1",
                    statements=["Statement 1", "Statement 2"],
                    severity=BiasSeverity.HIGH,
                    task_types=["task_a", "task_b"],
                )
            ],
        )

        data = policy.to_dict()
        assert data["name"] == "test"
        assert data["domain"] == "financial"
        assert len(data["dimensions"]) == 1
        assert data["dimensions"][0]["statements"] == ["Statement 1", "Statement 2"]

    def test_from_dict(self):
        """Should deserialize from dict."""
        original = CustomPolicy(
            name="test",
            display_name="Test",
            domain="healthcare",
            dimensions=[
                PolicyDimension(
                    name="dim1",
                    display_name="Dim 1",
                    statements=["Statement"],
                )
            ],
        )

        data = original.to_dict()
        restored = CustomPolicy.from_dict(data)

        assert restored.name == original.name
        assert restored.domain == original.domain
        assert len(restored.dimensions) == len(original.dimensions)


class TestTemplates:
    """Tests for policy templates."""

    def test_list_templates(self):
        """Should list all templates."""
        templates = list_templates()
        assert len(templates) >= 3  # At least financial, healthcare, hr

        names = {t["name"] for t in templates}
        assert "financial_services" in names
        assert "healthcare_equity" in names
        assert "hr_recruitment" in names

    def test_get_template(self):
        """Should get template by name."""
        policy = get_template("financial_services")

        assert policy.name == "financial_services"
        assert policy.domain == "financial"
        assert len(policy.dimensions) >= 2

    def test_get_unknown_template(self):
        """Should raise for unknown template."""
        with pytest.raises(ValueError) as exc_info:
            get_template("nonexistent")

        assert "Unknown template" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    def test_financial_template_structure(self):
        """Financial template should have expected structure."""
        policy = get_template("financial_services")

        dim_names = {d.name for d in policy.dimensions}
        assert "fair_lending" in dim_names

        fair_lending = next(d for d in policy.dimensions if d.name == "fair_lending")
        assert fair_lending.severity == BiasSeverity.CRITICAL
        assert len(fair_lending.statements) >= 3
        assert "loan_recommendation" in fair_lending.task_types

    def test_healthcare_template_structure(self):
        """Healthcare template should have expected structure."""
        policy = get_template("healthcare_equity")

        dim_names = {d.name for d in policy.dimensions}
        assert "diagnostic_equity" in dim_names
        assert "treatment_parity" in dim_names

    def test_hr_template_structure(self):
        """HR template should have expected structure."""
        policy = get_template("hr_recruitment")

        dim_names = {d.name for d in policy.dimensions}
        assert "hiring_fairness" in dim_names
        assert "compensation_equity" in dim_names


class TestDemographicCategory:
    """Tests for DemographicCategory."""

    def test_to_dict(self):
        """Should serialize to dict."""
        cat = DemographicCategory(
            id="test",
            name="Test",
            description="Test description",
            enabled=True,
            threshold=0.5,
            severity=BiasSeverity.HIGH,
            consensus=BiasConsensus.UNIVERSAL,
        )

        data = cat.to_dict()
        assert data["id"] == "test"
        assert data["severity"] == "high"
        assert data["consensus"] == "universal"

    def test_from_dict(self):
        """Should deserialize from dict."""
        data = {
            "id": "test",
            "name": "Test",
            "description": "Test description",
            "enabled": True,
            "threshold": 0.6,
            "severity": "medium",
            "consensus": "strong",
        }

        cat = DemographicCategory.from_dict(data)
        assert cat.id == "test"
        assert cat.threshold == 0.6
        assert cat.severity == BiasSeverity.MEDIUM
        assert cat.consensus == BiasConsensus.STRONG


class TestPolicyDimension:
    """Tests for PolicyDimension."""

    def test_default_values(self):
        """Should have sensible defaults."""
        dim = PolicyDimension(
            name="test",
            display_name="Test",
            statements=["Statement"],
        )

        assert dim.severity == BiasSeverity.MEDIUM
        assert dim.threshold == 0.5
        assert dim.task_types == ["*"]

    def test_custom_values(self):
        """Should accept custom values."""
        dim = PolicyDimension(
            name="test",
            display_name="Test",
            statements=["Statement 1", "Statement 2"],
            severity=BiasSeverity.CRITICAL,
            threshold=0.7,
            description="Custom description",
            task_types=["task1", "task2"],
        )

        assert dim.severity == BiasSeverity.CRITICAL
        assert dim.threshold == 0.7
        assert dim.description == "Custom description"
        assert dim.task_types == ["task1", "task2"]


class TestBiasEnums:
    """Tests for bias enums."""

    def test_severity_values(self):
        """Should have expected severity values."""
        assert BiasSeverity.CRITICAL.value == "critical"
        assert BiasSeverity.HIGH.value == "high"
        assert BiasSeverity.MEDIUM.value == "medium"
        assert BiasSeverity.LOW.value == "low"

    def test_consensus_values(self):
        """Should have expected consensus values."""
        assert BiasConsensus.UNIVERSAL.value == "universal"
        assert BiasConsensus.STRONG.value == "strong"
        assert BiasConsensus.MODERATE.value == "moderate"
        assert BiasConsensus.EMERGING.value == "emerging"
