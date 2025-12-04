"""
Composants UI réutilisables pour l'interface Streamlit.
"""

from src.front.components.chat_component import (
    AgentChatComponent,
    ChatComponent,
    ChatMessage,
    MessageRole,
    render_typing_indicator,
)
from src.front.components.results_component import (
    AnalysisResultsComponent,
    AutoMLResultsComponent,
    FEResultsComponent,
    MetricCard,
    PipelineResultsDashboard,
)

__all__ = [
    # Chat
    "ChatComponent",
    "AgentChatComponent",
    "ChatMessage",
    "MessageRole",
    "render_typing_indicator",
    # Results
    "MetricCard",
    "AnalysisResultsComponent",
    "FEResultsComponent",
    "AutoMLResultsComponent",
    "PipelineResultsDashboard",
]
