"""
Composants UI réutilisables pour l'interface Streamlit.
"""

from src.front.components.chat_component import (
    ChatComponent,
    AgentChatComponent,
    ChatMessage,
    MessageRole,
    render_typing_indicator,
)

from src.front.components.results_component import (
    MetricCard,
    AnalysisResultsComponent,
    FEResultsComponent,
    AutoMLResultsComponent,
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
