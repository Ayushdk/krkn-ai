from krkn_ai.models.cluster_components import (
    ClusterComponents,
    Namespace,
    Pod,
)
from krkn_ai.models.safety import SafetyConfig


def test_default_safety_blocks_kube_system():
    components = ClusterComponents(
        namespaces=[
            Namespace(
                name="kube-system",
                pods=[
                    Pod(
                        name="prometheus-k8s-0",
                        labels={"app": "prometheus"}
                    )
                ],
            ),
            Namespace(
                name="robot-shop",
                pods=[
                    Pod(
                        name="cart-123",
                        labels={"service": "cart"}
                    )
                ],
            ),
        ]
    )

    safety = SafetyConfig()

    components.apply_safety(safety)
    active = components.get_active_components()

    assert len(active.namespaces) == 1
    assert active.namespaces[0].name == "robot-shop"
