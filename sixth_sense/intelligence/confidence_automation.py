"""
Confidence-Aware Automation Engine — The Sixth Sense (SIH 2026 PS 26124 / PS 26125)
Enforces multi-tiered confidence gating and Human-In-The-Loop governance:

Automation Tiers:
  - LOW_CONFIDENCE (confidence < 0.40):
    Action: LOG_AND_GROUP (Monitored internally, no public work-order dispatched)
  - MEDIUM_CONFIDENCE (0.40 <= confidence < 0.65):
    Action: REVIEW_REQUIRED (Inspector verification queue)
  - HIGH_CONFIDENCE_UNCONFIRMED (confidence >= 0.65, single pass):
    Action: AWAITING_FLEET_CORROBORATION (Awaiting follow-up transit pass)
  - HIGH_CONFIDENCE_CORROBORATED (confidence >= 0.65, >= 2 bus passes):
    Action: ACTIONABLE_WORK_ORDER (Ready for public works prioritization)

High-Consequence Governance:
  - Legal, financial, or contractor penalty decisions are NEVER automatically binding.
  - System generates auditable review candidates requiring authorized municipal sign-off.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from sixth_sense.schemas.urban_event import (
    Observation,
    PersistentIssue,
)

logger = logging.getLogger(__name__)


class AutomationTier(str, Enum):
    LOG_AND_GROUP = "LOG_AND_GROUP"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    AWAITING_FLEET_CORROBORATION = "AWAITING_FLEET_CORROBORATION"
    ACTIONABLE_WORK_ORDER = "ACTIONABLE_WORK_ORDER"


class GovernanceAction(str, Enum):
    INTERNAL_MONITORING = "INTERNAL_MONITORING"
    SUPERVISOR_REVIEW_CANDIDATE = "SUPERVISOR_REVIEW_CANDIDATE"
    DISPATCHABLE_TASK = "DISPATCHABLE_TASK"
    HUMAN_APPROVAL_REQUIRED = "HUMAN_APPROVAL_REQUIRED"


@dataclass
class AutomationDecision:
    issue_id: str
    confidence: float
    bus_count: int
    observation_count: int
    tier: AutomationTier
    governance_action: GovernanceAction
    auto_dispatch_permitted: bool
    requires_human_signoff: bool
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "confidence": round(self.confidence, 4),
            "bus_count": self.bus_count,
            "observation_count": self.observation_count,
            "tier": self.tier.value,
            "governance_action": self.governance_action.value,
            "auto_dispatch_permitted": self.auto_dispatch_permitted,
            "requires_human_signoff": self.requires_human_signoff,
            "rationale": self.rationale,
        }


class ConfidenceAutomationGovernor:
    """
    Evaluates confidence-aware automation policies and enforces human-in-the-loop safeguards.
    """

    def __init__(
        self,
        low_threshold: float = 0.40,
        high_threshold: float = 0.65,
    ) -> None:
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def evaluate_issue(
        self,
        issue: PersistentIssue,
        is_high_consequence: bool = False,
    ) -> AutomationDecision:
        """
        Evaluate an issue for automated vs human-gated action.
        """
        conf = issue.confidence
        b_count = issue.bus_count
        o_count = issue.observation_count

        # High-consequence decisions (financial penalty, contractor dispute, legal notice)
        if is_high_consequence:
            return AutomationDecision(
                issue_id=issue.issue_id,
                confidence=conf,
                bus_count=b_count,
                observation_count=o_count,
                tier=AutomationTier.REVIEW_REQUIRED,
                governance_action=GovernanceAction.HUMAN_APPROVAL_REQUIRED,
                auto_dispatch_permitted=False,
                requires_human_signoff=True,
                rationale=(
                    "High-consequence legal/financial event; strict municipal policy dictates "
                    "mandatory human engineer verification before any binding action."
                ),
            )

        # Tier 1: Low confidence (< 0.40)
        if conf < self.low_threshold:
            return AutomationDecision(
                issue_id=issue.issue_id,
                confidence=conf,
                bus_count=b_count,
                observation_count=o_count,
                tier=AutomationTier.LOG_AND_GROUP,
                governance_action=GovernanceAction.INTERNAL_MONITORING,
                auto_dispatch_permitted=False,
                requires_human_signoff=False,
                rationale=f"Confidence {conf:.2f} is below automated threshold ({self.low_threshold}); logged for background temporal aggregation.",
            )

        # Tier 2: Medium confidence (0.40 <= conf < 0.65)
        if conf < self.high_threshold:
            return AutomationDecision(
                issue_id=issue.issue_id,
                confidence=conf,
                bus_count=b_count,
                observation_count=o_count,
                tier=AutomationTier.REVIEW_REQUIRED,
                governance_action=GovernanceAction.SUPERVISOR_REVIEW_CANDIDATE,
                auto_dispatch_permitted=False,
                requires_human_signoff=True,
                rationale=f"Moderate confidence {conf:.2f}; routed to supervisor review queue for visual confirmation.",
            )

        # Tier 3: High confidence (conf >= 0.65)
        if b_count >= 2:
            return AutomationDecision(
                issue_id=issue.issue_id,
                confidence=conf,
                bus_count=b_count,
                observation_count=o_count,
                tier=AutomationTier.ACTIONABLE_WORK_ORDER,
                governance_action=GovernanceAction.DISPATCHABLE_TASK,
                auto_dispatch_permitted=True,
                requires_human_signoff=False,
                rationale=f"High confidence ({conf:.2f}) corroborated across {b_count} multiple bus passes; eligible for automated municipal workflow creation (draft work order).",
            )
        else:
            return AutomationDecision(
                issue_id=issue.issue_id,
                confidence=conf,
                bus_count=b_count,
                observation_count=o_count,
                tier=AutomationTier.AWAITING_FLEET_CORROBORATION,
                governance_action=GovernanceAction.INTERNAL_MONITORING,
                auto_dispatch_permitted=False,
                requires_human_signoff=False,
                rationale=f"High confidence ({conf:.2f}) observed on single bus pass; awaiting multi-bus corroboration pass before automated workflow creation.",
            )
