#!/usr/bin/env python3
"""
CERT SDK: Bias Detection Examples

Demonstrates:
1. Default demographic bias detection
2. Custom demographic bias configuration
3. Using policy templates
4. Creating custom policies
5. Per-trace overrides
"""

from cert import CertClient
from cert.bias import (
    DemographicBiasConfig,
    CustomPolicy,
    PolicyDimension,
    BiasSeverity,
)

API_KEY = "your-api-key"
PROJECT = "bias-detection-demo"


def example_1_default_config():
    """Default configuration - demographic bias enabled with standard categories."""
    print("\n" + "=" * 60)
    print("Example 1: Default Configuration")
    print("=" * 60)

    # Default: demographic bias enabled, standard categories
    client = CertClient(api_key=API_KEY, project=PROJECT)

    # List what's enabled
    print("\nEnabled demographic categories:")
    for cat in client.list_demographic_categories():
        if cat["default_enabled"]:
            print(f"  [x] {cat['name']} ({cat['consensus']})")
        else:
            print(f"  [ ] {cat['name']} ({cat['consensus']})")

    # Send a trace - bias config is automatically included
    client.trace(
        provider="openai",
        model="gpt-4o",
        input_text="Describe a typical software engineer",
        output_text="Software engineers are analytical problem-solvers who design and build applications.",
        duration_ms=500,
    )
    print("\nTrace sent with default bias configuration")
    client.close()


def example_2_custom_demographic_config():
    """Customize demographic bias detection."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Demographic Configuration")
    print("=" * 60)

    # Create custom configuration
    config = DemographicBiasConfig()

    # Enable political bias (disabled by default)
    config.enable("political")

    # Stricter threshold for gender bias (lower = more sensitive)
    config.set_threshold("gender", 0.3)

    # Higher severity for age bias
    config.set_severity("age", BiasSeverity.HIGH)

    # Disable appearance bias
    config.disable("appearance")

    # Use with client
    client = CertClient(
        api_key=API_KEY,
        project=PROJECT,
        demographic_bias=config,
    )

    print("\nCustom configuration applied:")
    print("  - Political bias: ENABLED")
    print("  - Gender bias threshold: 0.3 (stricter)")
    print("  - Age bias severity: HIGH")
    client.close()


def example_3_policy_templates():
    """Use pre-built policy templates for domain-specific detection."""
    print("\n" + "=" * 60)
    print("Example 3: Policy Templates")
    print("=" * 60)

    # List available templates
    print("\nAvailable templates:")
    for tpl in CertClient.list_policy_templates():
        print(f"  - {tpl['display_name']} ({tpl['domain']}): {tpl['dimensions']} dimensions")

    # Use financial services template
    client = CertClient(
        api_key=API_KEY,
        project="loan-advisor",
        custom_policy="financial_services",
    )

    # Trace a loan decision with task_type for dimension filtering
    client.trace(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        input_text="Should we approve this loan application?",
        output_text="Based on the credit score of 720 and stable employment history, I recommend approval at standard rates.",
        duration_ms=1500,
        knowledge_base="Applicant: Credit Score 720, DTI 28%, Employment 5 years at current job",
        evaluation_mode="grounded",
        context_source="retrieval",
        task_type="loan_recommendation",  # Matches fair_lending dimension
    )
    print("\nTrace sent with financial_services policy")
    client.close()


def example_4_custom_policy():
    """Create a completely custom policy."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Policy")
    print("=" * 60)

    # Define custom policy for content moderation
    content_policy = CustomPolicy(
        name="content_guidelines",
        display_name="Content Guidelines",
        domain="general",
        description="Custom bias rules for content generation",
        dimensions=[
            PolicyDimension(
                name="stereotype_avoidance",
                display_name="Stereotype Avoidance",
                description="Avoid reinforcing stereotypes in generated content",
                severity=BiasSeverity.HIGH,
                threshold=0.5,
                statements=[
                    "Descriptions must not rely on demographic stereotypes.",
                    "Professional roles must not be associated with specific demographics.",
                    "Examples and scenarios must include diverse representation.",
                ],
                task_types=["content_generation", "creative_writing"],
            ),
            PolicyDimension(
                name="balanced_perspectives",
                display_name="Balanced Perspectives",
                description="Present balanced viewpoints on contested topics",
                severity=BiasSeverity.MEDIUM,
                threshold=0.4,
                statements=[
                    "Multiple perspectives must be acknowledged on contested topics.",
                    "Expert consensus must be accurately represented.",
                ],
                task_types=["analysis", "summary", "explanation"],
            ),
        ],
    )

    client = CertClient(
        api_key=API_KEY,
        project="content-generator",
        custom_policy=content_policy,
    )

    print(f"\nCustom policy '{content_policy.display_name}' applied")
    print(f"  Dimensions: {len(content_policy.dimensions)}")
    for dim in content_policy.dimensions:
        print(f"    - {dim.display_name}: {len(dim.statements)} statements")
    client.close()


def example_5_per_trace_overrides():
    """Override bias config for specific traces."""
    print("\n" + "=" * 60)
    print("Example 5: Per-Trace Overrides")
    print("=" * 60)

    # Default config
    client = CertClient(
        api_key=API_KEY,
        project=PROJECT,
        demographic_bias=True,
    )

    # Skip bias detection for a specific trace
    client.trace(
        provider="openai",
        model="gpt-4o",
        input_text="Internal test prompt",
        output_text="Internal test response",
        duration_ms=100,
        skip_bias_detection=True,  # No bias evaluation for this trace
    )
    print("\nTrace 1: Bias detection skipped")

    # Override demographic config for one trace
    strict_config = DemographicBiasConfig().enable_strict()
    client.trace(
        provider="openai",
        model="gpt-4o",
        input_text="User prompt",
        output_text="User response",
        duration_ms=200,
        demographic_bias_override=strict_config,  # Stricter for this trace
    )
    print("Trace 2: Strict demographic config override")

    # Add custom policy for one trace
    client.trace(
        provider="openai",
        model="gpt-4o",
        input_text="Healthcare question",
        output_text="Healthcare response",
        duration_ms=300,
        custom_policy_override="healthcare_equity",  # Add policy for this trace
        task_type="diagnosis",
    )
    print("Trace 3: Healthcare policy override")
    client.close()


def example_6_combined():
    """Combine demographic and custom policy detection."""
    print("\n" + "=" * 60)
    print("Example 6: Combined Configuration")
    print("=" * 60)

    # Both demographic bias AND custom policy
    config = DemographicBiasConfig()
    config.enable_standard()

    client = CertClient(
        api_key=API_KEY,
        project="healthcare-assistant",
        demographic_bias=config,
        custom_policy="healthcare_equity",
    )

    client.trace(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        input_text="Patient presents with chest pain and shortness of breath",
        output_text="Recommend immediate ECG, troponin levels, and cardiac workup given symptom presentation.",
        duration_ms=1500,
        knowledge_base="Patient: 55yo, BP 145/92, HR 88. History: Hypertension, family history of CAD.",
        evaluation_mode="grounded",
        context_source="retrieval",
        task_type="diagnosis",
    )

    print("\nTrace sent with:")
    print("  - Demographic bias: Standard categories")
    print("  - Custom policy: Healthcare equity")
    client.close()


def example_7_fluent_api():
    """Configure using fluent API."""
    print("\n" + "=" * 60)
    print("Example 7: Fluent API")
    print("=" * 60)

    client = (
        CertClient(api_key=API_KEY, project=PROJECT)
        .enable_demographic_category("political")
        .set_bias_threshold("gender", 0.4)
        .set_custom_policy("hr_recruitment")
    )

    print("\nClient configured via fluent API")
    print("  - Political bias: ENABLED")
    print("  - Gender threshold: 0.4")
    print("  - Custom policy: HR Recruitment")
    client.close()


def example_8_disable_bias():
    """Disable bias detection entirely."""
    print("\n" + "=" * 60)
    print("Example 8: Disable Bias Detection")
    print("=" * 60)

    # Disable at client level
    client = CertClient(
        api_key=API_KEY,
        project=PROJECT,
        demographic_bias=False,
        custom_policy=None,
    )

    print("\nBias detection disabled at client level")
    client.close()

    # Or disable after creation
    client2 = CertClient(api_key=API_KEY, project=PROJECT)
    client2.disable_bias_detection()
    print("Bias detection disabled via method")
    client2.close()


if __name__ == "__main__":
    example_1_default_config()
    example_2_custom_demographic_config()
    example_3_policy_templates()
    example_4_custom_policy()
    example_5_per_trace_overrides()
    example_6_combined()
    example_7_fluent_api()
    example_8_disable_bias()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
