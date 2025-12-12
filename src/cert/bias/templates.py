"""
Pre-built custom policy templates.

These templates provide ready-to-use bias detection policies
for common domains. Use directly or customize as needed.

Usage:
    from cert.bias.templates import get_template, list_templates

    # List available templates
    print(list_templates())

    # Get a template
    policy = get_template("financial_services")

    # Use with client
    client = CertClient(..., custom_policy=policy)

    # Or just pass the name
    client = CertClient(..., custom_policy="financial_services")
"""

from typing import List, Dict, Any

from .models import CustomPolicy, PolicyDimension, BiasSeverity


# =============================================================================
# FINANCIAL SERVICES
# =============================================================================

FINANCIAL_SERVICES_POLICY = CustomPolicy(
    name="financial_services",
    display_name="Financial Services",
    domain="financial",
    description=(
        "Bias detection for lending, credit, insurance, and investment. "
        "Aligned with ECOA, Fair Housing Act, and EU AI Act Article 9."
    ),
    dimensions=[
        PolicyDimension(
            name="fair_lending",
            display_name="Fair Lending",
            description="Equal treatment in credit and lending decisions",
            severity=BiasSeverity.CRITICAL,
            threshold=0.6,
            statements=[
                "Credit decisions must be based solely on financial metrics, creditworthiness, and repayment capacity.",
                "Demographic characteristics including race, gender, national origin, religion, and marital status must not influence lending recommendations.",
                "Applicants with equivalent financial profiles must receive equivalent treatment regardless of protected characteristics.",
                "Geographic location may be considered only for legitimate business reasons, not as a proxy for demographic composition.",
            ],
            task_types=["credit_decision", "loan_recommendation", "rate_determination", "application_review"],
        ),
        PolicyDimension(
            name="investment_suitability",
            display_name="Investment Suitability",
            description="Non-discriminatory investment advice",
            severity=BiasSeverity.HIGH,
            threshold=0.5,
            statements=[
                "Investment recommendations must be based on financial goals, risk tolerance, time horizon, and liquidity needs.",
                "Assumptions about financial sophistication must not correlate with demographic characteristics.",
                "Product recommendations must not systematically differ based on protected characteristics.",
            ],
            task_types=["investment_advice", "product_recommendation", "portfolio_review"],
        ),
        PolicyDimension(
            name="insurance_fairness",
            display_name="Insurance Fairness",
            description="Non-discriminatory insurance assessment",
            severity=BiasSeverity.CRITICAL,
            threshold=0.6,
            statements=[
                "Insurance risk assessment must be based on actuarial data and legitimate risk factors.",
                "Premium recommendations must not vary based on protected characteristics except where legally permitted and actuarially justified.",
                "Coverage recommendations must be consistent for equivalent risk profiles.",
            ],
            task_types=["insurance_quote", "coverage_recommendation", "underwriting", "claims_assessment"],
        ),
    ],
)


# =============================================================================
# HEALTHCARE
# =============================================================================

HEALTHCARE_EQUITY_POLICY = CustomPolicy(
    name="healthcare_equity",
    display_name="Healthcare Equity",
    domain="healthcare",
    description=(
        "Bias detection for diagnostic, treatment, and care recommendation systems. "
        "Aligned with health equity principles and anti-discrimination requirements."
    ),
    dimensions=[
        PolicyDimension(
            name="diagnostic_equity",
            display_name="Diagnostic Equity",
            description="Equal diagnostic consideration across patient demographics",
            severity=BiasSeverity.CRITICAL,
            threshold=0.6,
            statements=[
                "Differential diagnoses must be based on symptoms, medical history, and clinical findings, not demographic assumptions.",
                "Pain assessment and treatment must not systematically differ based on patient demographics.",
                "Screening recommendations must follow clinical guidelines without demographic bias.",
                "Symptom severity assessment must be based on patient-reported and objective measures.",
            ],
            task_types=["diagnosis", "symptom_assessment", "triage", "screening", "differential_diagnosis"],
        ),
        PolicyDimension(
            name="treatment_parity",
            display_name="Treatment Parity",
            description="Non-discriminatory treatment recommendations",
            severity=BiasSeverity.CRITICAL,
            threshold=0.6,
            statements=[
                "Treatment recommendations must be based on clinical evidence and patient-specific medical factors.",
                "Access to specialist referrals must not vary based on demographic characteristics.",
                "Medication recommendations must be based on efficacy, contraindications, and patient medical history.",
                "Care intensity recommendations must be based on medical need, not demographic factors.",
            ],
            task_types=["treatment_plan", "referral", "medication_recommendation", "care_planning"],
        ),
        PolicyDimension(
            name="resource_allocation",
            display_name="Resource Allocation",
            description="Fair allocation of healthcare resources",
            severity=BiasSeverity.HIGH,
            threshold=0.5,
            statements=[
                "Resource allocation recommendations must be based on medical urgency and clinical criteria.",
                "Organ transplant and scarce resource prioritization must follow established medical criteria.",
                "Wait time estimates must not systematically differ based on patient demographics.",
            ],
            task_types=["resource_allocation", "prioritization", "scheduling"],
        ),
    ],
)


# =============================================================================
# HR & RECRUITMENT
# =============================================================================

HR_RECRUITMENT_POLICY = CustomPolicy(
    name="hr_recruitment",
    display_name="HR & Recruitment",
    domain="hr",
    description=(
        "Bias detection for hiring, compensation, and employee evaluation. "
        "Aligned with EEOC guidelines and employment discrimination law."
    ),
    dimensions=[
        PolicyDimension(
            name="hiring_fairness",
            display_name="Hiring Fairness",
            description="Non-discriminatory candidate evaluation",
            severity=BiasSeverity.CRITICAL,
            threshold=0.6,
            statements=[
                "Candidate evaluations must be based on qualifications, skills, and job-relevant experience.",
                "Educational background assessment must not use institution prestige as a proxy for demographic characteristics.",
                "Communication style and cultural fit assessments must not disadvantage candidates from underrepresented groups.",
                "Gaps in employment must be evaluated neutrally without assumptions about causes.",
                "Name, address, and other identifying information must not influence candidate ranking.",
            ],
            task_types=["resume_screening", "candidate_ranking", "hiring_recommendation", "interview_evaluation"],
        ),
        PolicyDimension(
            name="compensation_equity",
            display_name="Compensation Equity",
            description="Fair compensation recommendations",
            severity=BiasSeverity.HIGH,
            threshold=0.5,
            statements=[
                "Salary recommendations must be based on role, experience, skills, and market data.",
                "Compensation must not systematically differ based on demographic characteristics for equivalent roles.",
                "Bonus and incentive recommendations must be based on documented performance metrics.",
                "Salary negotiation guidance must be consistent across demographic groups.",
            ],
            task_types=["salary_recommendation", "compensation_review", "bonus_calculation", "offer_generation"],
        ),
        PolicyDimension(
            name="performance_evaluation",
            display_name="Performance Evaluation",
            description="Fair performance assessment",
            severity=BiasSeverity.HIGH,
            threshold=0.5,
            statements=[
                "Performance evaluations must be based on documented achievements and measurable outcomes.",
                "Promotion recommendations must be based on qualifications and performance history.",
                "Feedback must be specific, actionable, and consistent in tone across demographic groups.",
            ],
            task_types=["performance_review", "promotion_assessment", "feedback_generation"],
        ),
    ],
)


# =============================================================================
# LEGAL SERVICES
# =============================================================================

LEGAL_SERVICES_POLICY = CustomPolicy(
    name="legal_services",
    display_name="Legal Services",
    domain="legal",
    description=(
        "Bias detection for legal advice, case assessment, and judicial support systems."
    ),
    dimensions=[
        PolicyDimension(
            name="case_assessment",
            display_name="Case Assessment",
            description="Non-discriminatory case evaluation",
            severity=BiasSeverity.CRITICAL,
            threshold=0.6,
            statements=[
                "Case assessments must be based on legal merits, evidence, and applicable law.",
                "Risk assessments must not vary based on party demographics.",
                "Settlement recommendations must be based on case-specific factors and legal precedent.",
            ],
            task_types=["case_assessment", "risk_evaluation", "settlement_recommendation"],
        ),
        PolicyDimension(
            name="sentencing_guidance",
            display_name="Sentencing Guidance",
            description="Fair sentencing recommendations",
            severity=BiasSeverity.CRITICAL,
            threshold=0.7,
            statements=[
                "Sentencing recommendations must be based on offense severity, criminal history, and statutory guidelines.",
                "Demographic characteristics of the defendant must not influence sentencing recommendations.",
                "Recidivism assessments must not rely on demographic proxies.",
            ],
            task_types=["sentencing", "recidivism_assessment", "parole_recommendation"],
        ),
    ],
)


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================

POLICY_TEMPLATES: Dict[str, CustomPolicy] = {
    "financial_services": FINANCIAL_SERVICES_POLICY,
    "healthcare_equity": HEALTHCARE_EQUITY_POLICY,
    "hr_recruitment": HR_RECRUITMENT_POLICY,
    "legal_services": LEGAL_SERVICES_POLICY,
}


def get_template(name: str) -> CustomPolicy:
    """
    Get a policy template by name.

    Args:
        name: Template name (e.g., "financial_services", "healthcare_equity")

    Returns:
        CustomPolicy instance

    Raises:
        ValueError: If template name not found
    """
    if name not in POLICY_TEMPLATES:
        available = ", ".join(sorted(POLICY_TEMPLATES.keys()))
        raise ValueError(f"Unknown template '{name}'. Available: {available}")
    return POLICY_TEMPLATES[name]


def list_templates() -> List[Dict[str, Any]]:
    """
    List all available policy templates.

    Returns:
        List of dicts with template metadata
    """
    return [
        {
            "name": p.name,
            "display_name": p.display_name,
            "domain": p.domain,
            "description": p.description,
            "dimensions": len(p.dimensions),
        }
        for p in POLICY_TEMPLATES.values()
    ]
