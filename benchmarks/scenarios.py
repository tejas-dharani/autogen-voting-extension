"""Benchmark scenarios for comparing voting vs standard group chat approaches."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ScenarioType(Enum):
    """Types of benchmark scenarios."""

    CODE_REVIEW = "code_review"
    ARCHITECTURE_DECISION = "architecture_decision"
    CONTENT_MODERATION = "content_moderation"
    MEDICAL_DIAGNOSIS = "medical_diagnosis"


@dataclass
class BenchmarkScenario:
    """Definition of a benchmark scenario for testing."""

    name: str
    scenario_type: ScenarioType
    description: str
    task_prompt: str
    expected_outcome: str | None = None
    success_criteria: dict[str, Any] = field(default_factory=dict)  # type: ignore
    agent_personas: list[dict[str, str]] = field(default_factory=list)  # type: ignore


# Code Review Scenarios
CODE_REVIEW_SCENARIOS = [
    BenchmarkScenario(
        name="bug_detection_security",
        scenario_type=ScenarioType.CODE_REVIEW,
        description="Security vulnerability detection in authentication code",
        task_prompt="""
Review this authentication code for security vulnerabilities and approve/reject for merge:

```python
def authenticate_user(username, password):
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    result = db.execute(query)
    if result:
        session['user_id'] = result[0]['id']
        return True
    return False
```

Decision: Should this code be approved for merge? Provide reasoning.
        """,
        expected_outcome="reject",
        success_criteria={
            "identifies_sql_injection": True,
            "identifies_plaintext_password": True,
            "provides_security_reasoning": True,
        },
        agent_personas=[
            {"name": "SecurityExpert", "role": "Security specialist focused on vulnerability detection"},
            {"name": "SeniorDev", "role": "Senior developer reviewing code quality and practices"},
            {"name": "TeamLead", "role": "Team lead balancing security with delivery timelines"},
        ],
    ),
    BenchmarkScenario(
        name="performance_optimization",
        scenario_type=ScenarioType.CODE_REVIEW,
        description="Performance review of data processing code",
        task_prompt="""
Review this data processing code for performance issues and approve/reject:

```python
def process_user_data(user_ids):
    results = []
    for user_id in user_ids:  # Could be 10,000+ users
        user = db.query(f"SELECT * FROM users WHERE id = {user_id}")
        profile = db.query(f"SELECT * FROM profiles WHERE user_id = {user_id}")
        stats = db.query(f"SELECT * FROM user_stats WHERE user_id = {user_id}")

        results.append({
            'user': user,
            'profile': profile,
            'stats': stats
        })
    return results
```

Decision: Should this code be approved for merge? Focus on performance implications.
        """,
        expected_outcome="reject",
        success_criteria={
            "identifies_n_plus_one": True,
            "suggests_batch_processing": True,
            "considers_scalability": True,
        },
        agent_personas=[
            {"name": "PerformanceEngineer", "role": "Performance engineer focused on scalability"},
            {"name": "DatabaseExpert", "role": "Database specialist reviewing query patterns"},
            {"name": "ProductionEngineer", "role": "Production engineer considering operational impact"},
        ],
    ),
    BenchmarkScenario(
        name="code_quality_readability",
        scenario_type=ScenarioType.CODE_REVIEW,
        description="Code quality and readability assessment",
        task_prompt="""
Review this function for code quality and readability:

```python
def calc(x, y, z, op):
    if op == 1:
        return x + y * z if z > 0 else x + y
    elif op == 2:
        return x - y * z if z > 0 else x - y
    elif op == 3:
        return x * y * z if z > 0 else x * y
    else:
        return x / y / z if z > 0 and y != 0 and z != 1 else x / y if y != 0 else 0
```

Decision: Should this code be approved? Consider maintainability and clarity.
        """,
        expected_outcome="reject",
        success_criteria={
            "identifies_poor_naming": True,
            "suggests_refactoring": True,
            "mentions_maintainability": True,
        },
        agent_personas=[
            {"name": "SeniorDev", "role": "Senior developer focused on code quality"},
            {"name": "CodeReviewer", "role": "Code reviewer specializing in maintainability"},
            {"name": "JuniorMentor", "role": "Mentor helping junior developers learn best practices"},
        ],
    ),
]


# Architecture Decision Scenarios
ARCHITECTURE_SCENARIOS = [
    BenchmarkScenario(
        name="microservices_vs_monolith",
        scenario_type=ScenarioType.ARCHITECTURE_DECISION,
        description="Decide between microservices and monolithic architecture",
        task_prompt="""
Our e-commerce platform needs to scale to handle 10x traffic growth. We're currently a monolith with:
- 50,000 daily active users
- 5-person development team
- Current response times: 200ms average
- Deployment frequency: Weekly releases

Options:
1. Break into microservices (user service, product service, order service, payment service)
2. Scale the monolith horizontally with better caching and database optimization
3. Hybrid approach: Extract only the most critical services

Decision: Which approach should we take? Consider team size, complexity, and growth projections.
        """,
        success_criteria={
            "considers_team_size": True,
            "addresses_complexity": True,
            "mentions_operational_overhead": True,
            "provides_concrete_recommendation": True,
        },
        agent_personas=[
            {"name": "SolutionArchitect", "role": "Solution architect with enterprise experience"},
            {"name": "TechLead", "role": "Technical lead familiar with current system"},
            {"name": "DevOpsEngineer", "role": "DevOps engineer considering operational complexity"},
        ],
    ),
    BenchmarkScenario(
        name="database_technology_choice",
        scenario_type=ScenarioType.ARCHITECTURE_DECISION,
        description="Choose database technology for analytics workload",
        task_prompt="""
We need to build a real-time analytics dashboard for our IoT platform processing:
- 1M events per minute
- Complex aggregations across time windows
- Sub-second query response requirements
- Mix of historical and real-time data

Options:
1. PostgreSQL with time-series extensions and read replicas
2. ClickHouse for analytical workloads
3. Apache Druid for real-time analytics
4. Hybrid: Kafka + ClickHouse + Redis

Decision: Which database architecture should we choose? Consider performance, complexity, and team expertise.
        """,
        success_criteria={
            "analyzes_performance_requirements": True,
            "considers_operational_complexity": True,
            "addresses_team_expertise": True,
            "provides_specific_recommendation": True,
        },
        agent_personas=[
            {"name": "DataArchitect", "role": "Data architect specializing in analytics systems"},
            {"name": "DatabaseEngineer", "role": "Database engineer with operational experience"},
            {"name": "TechLead", "role": "Technical lead considering team capabilities"},
        ],
    ),
]


# Content Moderation Scenarios
CONTENT_MODERATION_SCENARIOS = [
    BenchmarkScenario(
        name="community_post_moderation",
        scenario_type=ScenarioType.CONTENT_MODERATION,
        description="Moderate community forum post for policy violations",
        task_prompt="""
Moderate this community forum post:

"This new update is absolutely terrible! The developers clearly don't know what they're doing.
My workflow is completely broken now. I've been using this platform for 3 years and this is
the worst change yet. They should fire whoever made this decision and roll it back immediately.
Other users are saying the same thing - check the Discord where people are really speaking their minds about how incompetent the team is."

Decision: Should this post be:
1. Approved (no action)
2. Flagged for review
3. Removed for policy violation
4. Removed and user warned

Consider: constructive criticism vs. harassment, community guidelines, and user engagement.
        """,
        expected_outcome="flagged_for_review",
        success_criteria={
            "distinguishes_criticism_from_harassment": True,
            "considers_community_impact": True,
            "provides_moderation_reasoning": True,
        },
        agent_personas=[
            {"name": "CommunityManager", "role": "Community manager focused on positive environment"},
            {"name": "SafetySpecialist", "role": "Safety specialist detecting harmful content"},
            {"name": "LegalAdvisor", "role": "Legal advisor considering policy compliance"},
        ],
    ),
    BenchmarkScenario(
        name="technical_content_accuracy",
        scenario_type=ScenarioType.CONTENT_MODERATION,
        description="Moderate technical content for accuracy and helpfulness",
        task_prompt="""
Moderate this technical answer on our Q&A platform:

Question: "How do I securely store passwords in my web application?"

Answer: "Just use MD5 hashing - it's fast and secure enough for most applications.
Here's the code:

```python
import hashlib
def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()
```

MD5 is widely supported and has been around for years so it's proven reliable.
Some people say use bcrypt but that's overkill for simple applications."

Decision: Should this answer be:
1. Approved as helpful
2. Flagged for technical review
3. Removed for spreading misinformation
4. Edited with corrections

Consider: technical accuracy, security implications, and educational value.
        """,
        expected_outcome="removed_for_misinformation",
        success_criteria={
            "identifies_security_misinformation": True,
            "considers_educational_harm": True,
            "suggests_correct_approach": True,
        },
        agent_personas=[
            {"name": "SecurityExpert", "role": "Security expert reviewing technical accuracy"},
            {"name": "CommunityModerator", "role": "Community moderator balancing helpfulness and accuracy"},
            {"name": "TechnicalReviewer", "role": "Technical reviewer ensuring content quality"},
        ],
    ),
]


# Medical Diagnosis Scenarios
MEDICAL_DIAGNOSIS_SCENARIOS = [
    BenchmarkScenario(
        name="chest_pain_diagnosis",
        scenario_type=ScenarioType.MEDICAL_DIAGNOSIS,
        description="Multi-specialist consultation for chest pain diagnosis",
        task_prompt="""
Medical Case Consultation:

Patient: 55-year-old male presenting to emergency department
Chief Complaint: Severe chest pain for 2 hours
Vital Signs: BP 150/95, HR 102, RR 22, O2Sat 96% on room air
Symptoms: 
- Crushing substernal chest pain radiating to left arm
- Diaphoresis, nausea
- Pain started at rest, not relieved by position changes
- No recent trauma or fever

Diagnostic Tests:
- ECG: ST elevation in leads II, III, aVF
- Troponin I: Elevated at 8.5 ng/mL (normal <0.04)
- Chest X-ray: Clear lung fields, normal heart size
- Basic metabolic panel: Normal except glucose 180

History:
- Hypertension, diabetes type 2
- Former smoker (quit 5 years ago)
- Family history of heart disease
- Takes metformin, lisinopril

Consultation Question: What is the most likely diagnosis and recommended immediate treatment?

Options:
A) STEMI (ST-elevation myocardial infarction) - urgent cardiac catheterization
B) NSTEMI (Non-ST elevation MI) - medical management, delayed catheterization  
C) Unstable angina - medical management and monitoring
D) Aortic dissection - urgent CT angiography
E) Pulmonary embolism - anticoagulation and CT pulmonary angiogram

Provide your diagnosis vote with detailed medical reasoning.
        """,
        expected_outcome="A",  # STEMI with urgent intervention
        success_criteria={
            "correctly_identifies_stemi": True,
            "recommends_urgent_intervention": True,
            "provides_medical_reasoning": True,
            "considers_differential_diagnoses": True,
        },
        agent_personas=[
            {"name": "EmergencyPhysician", "role": "Emergency medicine physician with acute care expertise"},
            {"name": "Cardiologist", "role": "Interventional cardiologist specializing in heart conditions"},
            {"name": "Internist", "role": "Internal medicine physician with broad diagnostic experience"},
        ],
    ),
    BenchmarkScenario(
        name="pediatric_fever_assessment",
        scenario_type=ScenarioType.MEDICAL_DIAGNOSIS,
        description="Pediatric fever evaluation and management decision",
        task_prompt="""
Pediatric Case Consultation:

Patient: 18-month-old female brought by parents
Chief Complaint: Fever for 3 days, decreased appetite
Current Status:
- Temperature: 102.5°F (39.2°C) rectal
- HR 140, RR 28, alert but fussy
- No obvious source of infection on examination
- Eating less, decreased wet diapers
- No vomiting or diarrhea

Physical Examination:
- Well-appearing when not crying
- Clear lungs, no retractions
- No ear discharge or erythema
- Throat slightly red but no exudate
- No rash, no lymphadenopathy
- Soft abdomen, no organomegaly

Laboratory Results:
- WBC: 15,000 (elevated)
- Urinalysis: 2+ leukocyte esterase, 1+ nitrites, 20-50 WBC/hpf
- Urine culture: Pending (48 hours)

Clinical Question: What is the most appropriate immediate management?

Options:
A) Discharge home with fever management and close follow-up
B) Start oral antibiotics for suspected UTI and discharge
C) Admit for IV antibiotics and monitoring
D) Perform lumbar puncture to rule out meningitis
E) Obtain blood cultures and observe for 24 hours

Consider: Age-specific fever protocols, urinalysis findings, and patient stability.
        """,
        expected_outcome="B",  # UTI treatment with appropriate monitoring
        success_criteria={
            "identifies_likely_uti": True,
            "chooses_appropriate_antibiotics": True,
            "considers_age_specific_risks": True,
            "includes_follow_up_plan": True,
        },
        agent_personas=[
            {"name": "PediatricianER", "role": "Pediatric emergency medicine physician"},
            {"name": "GeneralPediatrician", "role": "General pediatrician with broad child health experience"},
            {"name": "PediatricNephrologist", "role": "Pediatric kidney specialist familiar with pediatric UTIs"},
        ],
    ),
    BenchmarkScenario(
        name="psychiatric_crisis_intervention",
        scenario_type=ScenarioType.MEDICAL_DIAGNOSIS,
        description="Psychiatric emergency assessment and safety determination",
        task_prompt="""
Psychiatric Emergency Consultation:

Patient: 28-year-old female brought by family
Chief Complaint: "Threatening to hurt herself" per family
Presentation:
- States "I can't take it anymore, I want to end the pain"
- Reports depression for 6 months after job loss and relationship breakup
- Stopped taking prescribed sertraline 2 weeks ago
- Drinking alcohol daily for past month
- Lives alone, limited social support

Mental Status Examination:
- Appears depressed, poor eye contact
- Speech: slow, soft, responds to questions
- Mood: "hopeless and worthless"
- Thought process: linear but with hopeless themes
- Reports passive death wish: "sometimes think I'd be better off dead"
- Denies active suicidal plan or intent currently
- No psychosis, no command auditory hallucinations
- Insight: acknowledges depression, wants help

Risk Factors:
- Recent major stressors
- Medication non-compliance
- Substance use
- Social isolation
- Previous depression history

Safety Assessment Question: What is the most appropriate disposition?

Options:
A) Voluntary psychiatric admission for safety and stabilization
B) Intensive outpatient program with daily check-ins and medication restart
C) Discharge home with family supervision and urgent psychiatry follow-up
D) Involuntary psychiatric hold due to imminent danger
E) Crisis respite care with peer support and medication management

Consider: Suicide risk assessment, protective factors, and treatment capacity.
        """,
        expected_outcome="C",  # Appropriate outpatient management with safety planning
        success_criteria={
            "assesses_suicide_risk_appropriately": True,
            "considers_protective_factors": True,
            "includes_safety_planning": True,
            "addresses_medication_compliance": True,
        },
        agent_personas=[
            {"name": "PsychiatristER", "role": "Emergency psychiatrist specializing in crisis intervention"},
            {"name": "ClinicalPsychologist", "role": "Clinical psychologist expert in suicide risk assessment"},
            {"name": "SocialWorker", "role": "Psychiatric social worker focused on discharge planning and safety"},
        ],
    ),
    BenchmarkScenario(
        name="radiological_mass_interpretation",
        scenario_type=ScenarioType.MEDICAL_DIAGNOSIS,
        description="Multidisciplinary radiology consultation for suspicious mass",
        task_prompt="""
Radiological Case Conference:

Patient: 62-year-old female with 6-month history of persistent cough
Clinical History:
- 30 pack-year smoking history (quit 2 years ago)
- Weight loss of 15 pounds over 4 months
- Intermittent hemoptysis (coughing up blood)
- No fever, night sweats, or chest pain

Imaging Findings:
CT Chest with Contrast:
- 3.2 cm spiculated mass in right upper lobe
- Multiple enlarged mediastinal lymph nodes (largest 2.1 cm)
- No pleural effusion
- No obvious metastatic disease in chest
- Liver and adrenals appear normal on current study

PET-CT Results:
- Hypermetabolic activity in lung mass (SUVmax 8.4)
- FDG uptake in mediastinal lymph nodes
- No other areas of abnormal uptake

Previous Imaging:
- Chest X-ray 8 months ago: reported as "clear"

Clinical Question: What is the most likely diagnosis and recommended next step?

Options:
A) Primary lung adenocarcinoma - proceed with tissue biopsy and staging
B) Inflammatory pseudotumor - follow-up imaging in 3 months
C) Metastatic disease from unknown primary - full body staging
D) Infectious process (atypical pneumonia) - antibiotic trial
E) Sarcoidosis with pulmonary involvement - bronchoscopy with biopsy

Consider: Imaging characteristics, clinical presentation, and smoking history.
        """,
        expected_outcome="A",  # Primary lung cancer requiring tissue diagnosis
        success_criteria={
            "correctly_identifies_malignant_features": True,
            "recommends_appropriate_biopsy": True,
            "considers_staging_requirements": True,
            "integrates_clinical_and_imaging_data": True,
        },
        agent_personas=[
            {"name": "RadiologistThoracic", "role": "Thoracic radiologist specializing in chest imaging"},
            {"name": "PulmonologistOnc", "role": "Pulmonologist with oncology experience"},
            {"name": "OncologistThoracic", "role": "Medical oncologist specializing in lung cancer"},
        ],
    ),
]


# Combined scenario collections
ALL_SCENARIOS = {
    ScenarioType.CODE_REVIEW: CODE_REVIEW_SCENARIOS,
    ScenarioType.ARCHITECTURE_DECISION: ARCHITECTURE_SCENARIOS,
    ScenarioType.CONTENT_MODERATION: CONTENT_MODERATION_SCENARIOS,
    ScenarioType.MEDICAL_DIAGNOSIS: MEDICAL_DIAGNOSIS_SCENARIOS,
}


def get_scenarios_by_type(scenario_type: ScenarioType) -> list[BenchmarkScenario]:
    """Get all scenarios of a specific type."""
    return ALL_SCENARIOS.get(scenario_type, [])


def get_all_scenarios() -> list[BenchmarkScenario]:
    """Get all available benchmark scenarios."""
    scenarios: list[BenchmarkScenario] = []
    for scenario_list in ALL_SCENARIOS.values():
        scenarios.extend(scenario_list)
    return scenarios


def get_scenario_by_name(name: str) -> BenchmarkScenario | None:
    """Get a specific scenario by name."""
    for scenario in get_all_scenarios():
        if scenario.name == name:
            return scenario
    return None
