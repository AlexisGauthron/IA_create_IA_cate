"""
Composant de chat réutilisable pour l'agent métier.
"""

import streamlit as st
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum


class MessageRole(Enum):
    """Rôles des messages dans le chat."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ChatMessage:
    """Structure d'un message de chat."""
    role: MessageRole
    content: str
    metadata: Optional[Dict[str, Any]] = None


class ChatComponent:
    """
    Composant de chat réutilisable pour Streamlit.

    Usage:
    ```python
    chat = ChatComponent(
        session_key="my_chat",
        on_send=my_callback_function,
    )
    chat.render()
    ```
    """

    def __init__(
        self,
        session_key: str = "chat_history",
        on_send: Optional[Callable[[str], Dict[str, Any]]] = None,
        placeholder: str = "Votre message...",
        assistant_avatar: str = "🤖",
        user_avatar: str = "👤",
        system_avatar: str = "⚙️",
        commands: Optional[Dict[str, str]] = None,
    ):
        """
        Initialise le composant chat.

        Args:
            session_key: Clé pour stocker l'historique en session state
            on_send: Callback appelé quand l'utilisateur envoie un message
            placeholder: Texte placeholder de l'input
            assistant_avatar: Avatar pour les messages assistant
            user_avatar: Avatar pour les messages utilisateur
            system_avatar: Avatar pour les messages système
            commands: Dictionnaire des commandes spéciales {commande: description}
        """
        self.session_key = session_key
        self.on_send = on_send
        self.placeholder = placeholder
        self.assistant_avatar = assistant_avatar
        self.user_avatar = user_avatar
        self.system_avatar = system_avatar
        self.commands = commands or {
            "skip": "Passer la question",
            "done": "Terminer la conversation",
        }

        # Initialiser l'historique si nécessaire
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = []

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Retourne l'historique des messages."""
        return st.session_state.get(self.session_key, [])

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Ajoute un message à l'historique."""
        message = {
            "role": role,
            "content": content,
            "metadata": metadata or {},
        }
        st.session_state[self.session_key].append(message)

    def clear_history(self):
        """Efface l'historique."""
        st.session_state[self.session_key] = []

    def render_commands_help(self):
        """Affiche l'aide des commandes."""
        if self.commands:
            commands_text = " | ".join([
                f"`{cmd}` ({desc})" for cmd, desc in self.commands.items()
            ])
            st.caption(f"Commandes: {commands_text}")

    def render_message(self, message: Dict[str, Any]):
        """Affiche un message."""
        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "assistant":
            avatar = self.assistant_avatar
        elif role == "system":
            avatar = self.system_avatar
        else:
            avatar = self.user_avatar

        with st.chat_message(role, avatar=avatar):
            st.markdown(content)

            # Afficher les métadonnées si présentes
            metadata = message.get("metadata", {})
            if metadata.get("mode"):
                st.caption(f"Mode: {metadata['mode']}")

    def render_history(self):
        """Affiche l'historique des messages."""
        for message in self.history:
            self.render_message(message)

    def process_input(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Traite l'input utilisateur.

        Returns:
            Résultat du callback ou None si commande spéciale
        """
        normalized_input = user_input.lower().strip()

        # Vérifier les commandes spéciales
        if normalized_input in ["skip", "passer"]:
            self.add_message("user", user_input)
            self.add_message(
                "assistant",
                "Question passée.",
                {"mode": "Skip"}
            )
            return {"action": "skip"}

        if normalized_input in ["done", "terminé", "fini"]:
            self.add_message("user", user_input)
            self.add_message(
                "assistant",
                "Conversation terminée. Les informations ont été enregistrées.",
                {"mode": "Final"}
            )
            return {"action": "done"}

        # Ajouter le message utilisateur
        self.add_message("user", user_input)

        # Appeler le callback si défini
        if self.on_send:
            try:
                response = self.on_send(user_input)

                # Ajouter la réponse à l'historique
                if isinstance(response, dict):
                    content = response.get(
                        "question",
                        response.get("synthesis", response.get("content", str(response)))
                    )
                    mode = response.get("mode", "Question")
                else:
                    content = str(response)
                    mode = "Question"

                self.add_message("assistant", content, {"mode": mode})
                return response

            except Exception as e:
                error_msg = f"Erreur: {str(e)}"
                self.add_message("system", error_msg)
                return {"error": str(e)}

        return None

    def render(self, show_commands: bool = True):
        """
        Rendu complet du composant chat.

        Args:
            show_commands: Afficher l'aide des commandes
        """
        # Aide des commandes
        if show_commands:
            self.render_commands_help()

        # Zone de chat scrollable
        chat_container = st.container()

        with chat_container:
            self.render_history()

        # Input utilisateur
        if user_input := st.chat_input(self.placeholder):
            result = self.process_input(user_input)

            # Gérer les actions spéciales
            if result:
                action = result.get("action")
                if action == "done":
                    st.session_state["chat_done"] = True

                # Rerun pour afficher le nouveau message
                st.rerun()


class AgentChatComponent(ChatComponent):
    """
    Composant chat spécialisé pour l'agent métier.
    Gère automatiquement la connexion avec BusinessClarificationBot.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        stats_report: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialise le chat avec l'agent métier.

        Args:
            provider: Provider LLM (openai, ollama)
            model: Modèle LLM
            stats_report: Rapport statistique pour initialiser la conversation
        """
        self.provider = provider
        self.model = model
        self.stats_report = stats_report
        self._chatbot = None

        super().__init__(
            session_key="agent_chat_history",
            on_send=self._handle_message,
            placeholder="Répondez à l'agent...",
            **kwargs
        )

    @property
    def chatbot(self):
        """Lazy loading du chatbot."""
        if self._chatbot is None:
            chatbot_key = "metier_chatbot"
            if chatbot_key in st.session_state:
                self._chatbot = st.session_state[chatbot_key]
            else:
                from src.analyse.metier.chatbot_llm import BusinessClarificationBot
                from src.core.llm_client import OllamaClient

                # Créer le client LLM (OllamaClient gère les deux providers)
                llm_client = OllamaClient(
                    model=self.model,
                    provider=self.provider,  # "openai" ou "ollama"
                )

                # Créer le chatbot
                self._chatbot = BusinessClarificationBot(
                    stats=self.stats_report or {},
                    llm=llm_client,
                )
                st.session_state[chatbot_key] = self._chatbot
        return self._chatbot

    def start_conversation(self) -> Dict[str, Any]:
        """Démarre la conversation avec le rapport statistique."""
        if not self.stats_report:
            return {"error": "Pas de rapport statistique fourni"}

        try:
            # ask_next(None) pour le premier appel
            response = self.chatbot.ask_next(user_answer=None)

            self.add_message("assistant", response, {"mode": "Question"})

            return {"content": response}

        except Exception as e:
            error_msg = f"Erreur au démarrage: {str(e)}"
            self.add_message("system", error_msg)
            return {"error": str(e)}

    def _handle_message(self, user_input: str) -> Dict[str, Any]:
        """Gère l'envoi d'un message au chatbot."""
        # ask_next retourne une string
        response = self.chatbot.ask_next(user_answer=user_input)
        return {"content": response}

    def get_context(self) -> Dict[str, Any]:
        """Récupère le contexte collecté par l'agent (historique des messages)."""
        if hasattr(self.chatbot, 'messages'):
            return {"messages": self.chatbot.messages}
        return {}

    def render_with_start_button(self):
        """Rendu avec bouton de démarrage si conversation vide."""
        if not self.history:
            st.markdown("""
            L'agent va analyser votre dataset et vous poser des questions
            pour mieux comprendre votre problème métier.
            """)

            if st.button("🚀 Démarrer la conversation", type="primary"):
                self.start_conversation()
                st.rerun()
        else:
            self.render()


def render_typing_indicator():
    """Affiche un indicateur de frappe."""
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 8px; padding: 8px;">
        <span style="font-size: 1.2rem;">🤖</span>
        <span style="color: var(--text-muted);">L'agent réfléchit...</span>
        <div class="typing-dots">
            <span>.</span><span>.</span><span>.</span>
        </div>
    </div>
    <style>
    .typing-dots span {
        animation: blink 1.4s infinite both;
        font-size: 1.5rem;
    }
    .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes blink {
        0%, 80%, 100% { opacity: 0; }
        40% { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)
