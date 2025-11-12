from typing import Dict, List, Optional
import json, re, random
from collections import Counter

def definition_labels_concises(
    shots: Dict[str, List[str]],
    *,
    model: str = "mistral:latest",   # ex: "llama3.1:8b", "mistral:latest"
    max_pos_examples: int = 12,      # nb max d'exemples positifs envoyés par label
    max_neg_examples: int = 10,      # nb max d'exemples négatifs (autres labels) par label
    max_terms: int = 12,             # nb max de termes demandés au LLM
    temperature: float = 0.0,
    seed: Optional[int] = 42,
) -> Dict[str, str]:
    """
    Génère des définitions courtes et discriminantes pour chaque label, au format:
        { "Support Technique": "Problèmes techniques, bugs, erreurs, ...", ... }

    - Utilise Ollama si disponible (chat → JSON dict {label: "liste, de, termes"}).
    - Fournit au LLM quelques exemples positifs / négatifs + candidats de termes.
    - Fallback (si Ollama absent): top termes discriminants calculés localement.
    """
    # ---------------- Helpers: nettoyage & tokenisation ----------------
    def _clean_list(xs: List[str]) -> List[str]:
        seen, out = set(), []
        for s in xs or []:
            if s is None: 
                continue
            s2 = str(s).strip()
            if not s2 or s2 in seen:
                continue
            seen.add(s2)
            out.append(s2[:400])  # coupe un peu pour éviter un prompt énorme
        return out

    tok_re = re.compile(r"(?u)\b[\w\-]+\b", re.UNICODE)
    stop_fr = {
        "le","la","les","de","des","du","un","une","et","ou","à","a","au","aux",
        "en","pour","par","sur","dans","avec","sans","ce","cet","cette","ces",
        "est","sont","être","avoir","fait","faire","plus","moins","très","tres",
        "bonjour","merci","bonjour,", "svp","s'il","sil","vous","je","nous","vous",
        "ils","elles","devis","numéro","numero","date"  # ajoute si besoin
    }

    def _tokens(s: str) -> List[str]:
        toks = [t.lower() for t in tok_re.findall(s)]
        toks = [t for t in toks if len(t) >= 3 and not t.isdigit() and t not in stop_fr]
        return toks

    # ---------------- Stats corpus pour fallback & candidats ----------------
    rng = random.Random(1234)
    labels = list(shots.keys())
    clean_shots = {lbl: _clean_list(exs) for lbl, exs in shots.items()}

    per_label_counts: Dict[str, Counter] = {}
    total_counts = Counter()
    for lbl, exs in clean_shots.items():
        c = Counter()
        for e in exs:
            c.update(_tokens(e))
        per_label_counts[lbl] = c
        total_counts.update(c)

    # helper: calcule top termes discriminants pos vs neg
    def _top_terms(lbl: str, k: int = 30) -> List[str]:
        pos = per_label_counts.get(lbl, Counter())
        neg = Counter()
        for other, cnts in per_label_counts.items():
            if other == lbl: 
                continue
            neg.update(cnts)
        # score de spécificité lissé
        def spec(t): 
            p, n = pos[t], neg[t]
            return (p + 1.0) / (p + n + 2.0)
        candidates = [t for t, cnt in pos.most_common(3*k) if cnt >= 2]
        candidates.sort(key=lambda t: (spec(t), pos[t]), reverse=True)
        return candidates[:k]

    # ---------------- Prompt LLM ----------------
    def _ollama_available():
        try:
            import ollama  # noqa
            return True
        except Exception:
            return False

    def _ask_ollama(payload: str) -> Optional[dict]:
        try:
            import ollama
            resp = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": (
                        "Tu es un assistant de taxonomie. "
                        "Objectif: produire, pour chaque label, UNE courte liste de mots/expressions séparés par des virgules, "
                        "clairement discriminants. "
                        "Langue: français. Réponds UNIQUEMENT avec un JSON valide."
                    )},
                    {"role": "user", "content": payload},
                ],
                options={
                    "temperature": float(temperature),
                    **({"seed": int(seed)} if seed is not None else {}),
                },
            )
            txt = resp["message"]["content"]
        except Exception:
            return None

        # Extraction JSON tolérante
        m = re.search(r"\{[\s\S]*\}", txt)
        if not m:
            return None
        raw = m.group(0)
        try:
            return json.loads(raw)
        except Exception:
            raw2 = re.sub(r",\s*([}\]])", r"\1", raw)
            try:
                return json.loads(raw2)
            except Exception:
                return None

    # Prépare un seul prompt global pour réduire les tours
    # Pour chaque label: exemples + candidats (pos) + négatifs échantillonnés
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

        candidates = _top_terms(lbl, k=max_terms * 2)  # un peu plus que max_terms
        lines.append(
            f"Label: {lbl}\n"
            f"- Exemples positifs:\n  - " + "\n  - ".join(pos) + "\n"
            f"- Exemples négatifs (autres labels):\n  - " + ("\n  - ".join(neg) if neg else "(aucun)") + "\n"
            f"- Candidats (termes discriminants côté client): {candidates}\n"
        )

    instruction = (
        "À partir des blocs ci-dessus, produis un objet JSON dont CHAQUE clé est exactement un des labels fournis, "
        f"et dont la valeur est une CHAÎNE en français listant {max(6, min(12, max_terms))} à {max_terms} "
        "mots/expressions COURTS, séparés par des virgules, sans point final, sans phrases.\n"
        "Règles:\n"
        "- Priorité aux termes discriminants (ce qui distingue ce label des autres),\n"
        "- Préfère noms/expressions stables (entités, thèmes, actions),\n"
        "- Évite les mots vides et le bruit (formules de politesse, etc.),\n"
        "- Pas de texte en dehors du JSON demandé."
    )
    payload = "\n\n".join(lines) + "\n\n" + instruction

    results: Dict[str, str] = {}

    if _ollama_available():
        data = _ask_ollama(payload)
        if isinstance(data, dict):
            # Post-traitement léger: normalise espaces, coupe longueur
            for lbl in labels:
                val = data.get(lbl)
                if isinstance(val, str):
                    s = re.sub(r"\s*,\s*", ", ", val.strip())
                    s = re.sub(r"\s+", " ", s)
                    s = s.rstrip(".;: ")
                    results[lbl] = s[:300]
        # Si le LLM n'a pas renvoyé tous les labels, on complémente via fallback ci-dessous

    # ---------------- Fallback / Complément local ----------------
    for lbl in labels:
        if lbl in results and results[lbl]:
            continue
        # construit une liste locale de termes discriminants
        terms = _top_terms(lbl, k=max_terms)
        # si insuffisant, complète avec termes fréquents de la classe
        if len(terms) < max_terms:
            extra = [t for t, _ in per_label_counts[lbl].most_common(max_terms*2) if t not in terms]
            terms.extend(extra[: max(0, max_terms - len(terms))])
        if not terms:
            # dernière chance: quelques n-grams naïfs à partir d'exemples
            sample = clean_shots.get(lbl, [])[:5]
            if sample:
                terms = [w for w in _tokens(" ".join(sample))][:max_terms]
        results[lbl] = ", ".join(terms[:max_terms]) if terms else "—"

    return results
