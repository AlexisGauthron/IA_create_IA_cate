# Module Core

Module fondamental : configuration centralisée, client LLM unifié, I/O données, gestion des chemins.

📚 **Documentation complète** : [docs/modules/core.md](../../docs/modules/core.md)

## Fichiers

| Fichier | Rôle |
|---------|------|
| `config.py` | Singleton `settings` - clés API, chemins |
| `base_path_config.py` | Classe abstraite pour gestion des outputs |
| `llm_client.py` | Client unifié OpenAI/Ollama |
| `io_utils.py` | Chargement datasets (CSV, sklearn) |
| `preprocessing.py` | Split train/test |
| `dataframe_utils.py` | Utilitaires DataFrame |
| `text_cleaning.py` | Nettoyage texte |
