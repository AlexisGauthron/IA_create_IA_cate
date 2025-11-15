from typing import Dict, List, Optional
import json, re, random
from collections import Counter

def definition_labels_completes(
    shots: Dict[str, List[str]],
    *,
    model: str = "llama2:latest",
    max_pos_examples: int = 8,
    max_neg_examples: int = 6,
    max_terms: int = 12,
    temperature: float = 0.4,
    seed: Optional[int] = 42,
    batch_size: int = 5,  # Nombre de labels par lot pour les appels LLM
) -> Dict[str, str]:
    """
    Génère pour chaque label une définition concrète + exemples extrapolés.
    Retourne un dict du type :
        {
          "Support Technique": 
             "Catégorie liée à la résolution de problèmes techniques, bugs ou erreurs. 
              Exemples : assistance aux utilisateurs, diagnostic de pannes, correction d’anomalies."
        }

    Utilise Ollama si disponible, sinon un fallback local basé sur les mots discriminants.
    """
    import src.fonctions.clean_tokeniser as clean_tok

    rng = random.Random(seed or 1234)
    labels = list(shots.keys())

    

    # Nettoyage des exemples
    clean_shots = {lbl: clean_tok._clean_list(exs) for lbl, exs in shots.items()}

    # Comptage global et par label
    per_label_counts: Dict[str, Counter] = {}
    total_counts = Counter()
    for lbl, exs in clean_shots.items():
        c = Counter()
        for e in exs:
            c.update(clean_tok._tokens(e))
        per_label_counts[lbl] = c
        total_counts.update(c)

    

    # --- Fonction interne : top termes discriminants
    def _top_terms(lbl: str, k: int = 30) -> List[str]:
        pos = per_label_counts.get(lbl, Counter())
        neg = Counter()
        for other, cnts in per_label_counts.items():
            if other == lbl:
                continue
            neg.update(cnts)
        def spec(t):
            p, n = pos[t], neg[t]
            return (p + 1.0) / (p + n + 2.0)
        candidates = [t for t, cnt in pos.most_common(3*k) if cnt >= 2]
        candidates.sort(key=lambda t: (spec(t), pos[t]), reverse=True)
        return candidates[:k]


    # --- Vérifie la disponibilité d’Ollama
    def _ollama_available():
        try:
            import ollama
            print("\n[get_definition] Ollama disponible pour génération LLM.\n")
            return True
        except Exception:
            print("\n[get_definition] Ollama indisponible, fallback local.\n")
            return False



    # --- Envoie la requête LLM (avec format JSON forcé)
    def _ask_ollama(payload: str) -> Optional[dict]:
        try:
            import ollama
            resp = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tu es un assistant expert en taxonomie métier. "
                            "Ton objectif est d'aider à améliorer le prototypage d'un système de classification automatique de fonctions métiers. "
                            "Ta réponse doit être STRICTEMENT un objet JSON valide, sans explication ni texte hors JSON. "
                            "Chaque clé du JSON correspond à un label, et chaque valeur est une phrase courte définissant concrètement la catégorie, "
                            "suivie de 2 à 3 exemples extrapolés représentatifs. "
                            "Utilise un ton neutre et concis, adapté à une base de données métier."
                        )

                    },
                    {
                        "role": "user",
                        "content": payload,
                    },
                ],
                options={
                    "temperature": float(temperature),
                    **({"seed": int(seed)} if seed is not None else {}),
                },
                format = "json",
            )
            print(f"[get_definition] Réponse reçue de l’LLM : \n {resp}\n\n\n")
            txt = resp["message"]["content"]


            # --- ⚙️ Post-traitement : extraire le JSON pur
            m = re.search(r"\{[\s\S]*\}", txt)
            if not m:
                print("[get_definition] ❌ Aucune structure JSON trouvée dans la réponse.\n")
                print("Réponse brute :\n", txt[:500], "\n---\n")
                return None
            raw = m.group(0)

            try:
                data = json.loads(raw)
                return data
            except json.JSONDecodeError:
                print("[get_definition] ⚠️ JSON invalide, tentative de correction...\n")
                raw2 = re.sub(r",\s*([}\]])", r"\1", raw)
                try:
                    return json.loads(raw2)
                except Exception as e:
                    print(f"[get_definition] Erreur JSON finale : {e}\n")
                    print("Texte reçu :\n", raw[:500], "\n---\n")
                    return None

        except Exception as e:
            import traceback
            print("[get_definition] Erreur critique lors de la requête Ollama :")
            traceback.print_exc()
            print(f"→ Détail : {e}\n")
            return None


    # --- Préparation du prompt global 
    lines = []
    for lbl in labels:
        pos = clean_shots.get(lbl, [])
        if len(pos) > max_pos_examples:
            pos = rng.sample(pos, max_pos_examples)
        neg_pool = []
        for other in labels:
            if other == lbl:
                continue
            neg_pool.extend(clean_shots.get(other, []))
        if len(neg_pool) > max_neg_examples:
            neg = rng.sample(neg_pool, max_neg_examples)
        else:
            neg = neg_pool

        candidates = _top_terms(lbl, k=max_terms)
        lines.append(
            f"Label: {lbl}\n"
            f"- Exemples positifs:\n  - " + "\n  - ".join(pos) + "\n"
            f"- Exemples négatifs:\n  - " + ("\n  - ".join(neg) if neg else "(aucun)") + "\n"
            f"- Termes clés pertinents: {candidates}\n"
        )

    instruction = (
        "À partir des blocs ci-dessus, produis un objet JSON dont chaque clé est un label, "
        "et chaque valeur est UNE phrase concise décrivant ce que représente cette catégorie, "
        "suivie d'exemples extrapolés cohérents (en français). "
        "Exemple de format attendu:\n"
        "{ \"Support Technique\": \"Aide aux utilisateurs confrontés à des problèmes logiciels ou matériels. "
        "Exemples : assistance informatique, résolution de bugs, diagnostic réseau.\" }\n"
        "Ne produis rien d’autre que ce JSON."
    )

    results: Dict[str, str] = {}

    # Appels par batchs (5 labels max par requête)
    if _ollama_available():
        all_data = {}
        for i in range(0, len(labels), batch_size):
            batch = labels[i : i + batch_size]
            print(f"[get_definition] Envoi du lot {i//batch_size + 1} "
                  f"({len(batch)} labels: {batch}) \n")

            # ⚙️ MODIF — Filtrer les lignes pour ce lot uniquement
            batch_lines = [l for l in lines if any(lbl in l for lbl in batch)]
            batch_payload = "\n\n".join(batch_lines) + "\n\n" + instruction

            try:
                data = _ask_ollama(batch_payload)
            except Exception as e:
                import traceback
                print("[get_definition] Erreur lors du traitement d’un lot :")
                traceback.print_exc()
                print(f"→ Détail : {e}\n")
                continue

            if isinstance(data, dict):
                all_data.update(data)
            else:
                print(f"[get_definition] ⚠️ Le modèle n’a pas renvoyé de JSON valide pour le lot {i//batch_size + 1}.\n")

        # ⚙️ MODIF — Nettoyage final des résultats fusionnés

        def normalize_label(lbl: str) -> str:
            return re.sub(r"\s+", " ", lbl.strip()).lower()


        # Crée un dict normalisé du JSON renvoyé
        normalized_data = {normalize_label(k): v for k, v in data.items()}

        # Récupération
        for lbl in labels:
            val = normalized_data.get(normalize_label(lbl))
            if isinstance(val, str):
                s = re.sub(r"\s+", " ", val.strip())
                s = s.rstrip(".;: ")
                results[lbl] = s[:500]


        print(f"[get_definition] Définitions générées via LLM pour {len(results)}/{len(labels)} labels.\n")

    else:
        print("[get_definition] Ollama indisponible, fallback local.\n")

    # --- Fallback local (si Ollama indisponible)
    for lbl in labels:
        if lbl in results and results[lbl]:
            continue
        terms = _top_terms(lbl, k=max_terms)
        exemples = clean_shots.get(lbl, [])[:3]
        definition = (
            f"Catégorie associée à {', '.join(terms[:max_terms])}. "
            f"Exemples typiques : {', '.join(exemples)}."
            if terms else
            f"Catégorie {lbl} (définition locale non disponible)."
        )
        results[lbl] = definition

    return results
