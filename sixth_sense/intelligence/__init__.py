"""
Urban Intelligence module — The Sixth Sense (SIH 2026 PS 26124 / PS 26125)
Multi-domain intelligence layer supporting Person 1 (Road) and Person 2 (Traffic).
"""
from sixth_sense.intelligence.road_health import (
    RoadHealthEngine,
    RoadSegmentHealth,
    RoadHealthState,
    RoadTrend,
    apply_closure_to_road_health,
)
from sixth_sense.intelligence.cross_domain_fusion import (
    CrossDomainFusionEngine,
    FusedSegmentContext,
)
from sixth_sense.intelligence.confidence_automation import (
    ConfidenceAutomationGovernor,
    AutomationTier,
    GovernanceAction,
    AutomationDecision,
)

__all__ = [
    "RoadHealthEngine",
    "RoadSegmentHealth",
    "RoadHealthState",
    "RoadTrend",
    "apply_closure_to_road_health",
    "CrossDomainFusionEngine",
    "FusedSegmentContext",
    "ConfidenceAutomationGovernor",
    "AutomationTier",
    "GovernanceAction",
    "AutomationDecision",
]
