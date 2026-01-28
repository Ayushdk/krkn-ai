from typing import List
from pydantic import BaseModel


class SafetyConfig(BaseModel):
    """
    Safety guardrails to prevent Krkn-AI from disrupting
    services it depends on (Prometheus, monitoring, control-plane).
    """

    excluded_namespaces: List[str] = [
        "kube-system",
        "kube-public",
        "kube-node-lease",
    ]

    excluded_pod_labels: List[str] = []

    excluded_pod_name_patterns: List[str] = []

    excluded_node_labels: List[str] = []
