## *Metrique Evaluation*

<br>

# 📊 Métriques de classification


## ***F1 (F-Score F1)*** — moyenne harmonique

> **F1** est la *moyenne harmonique* entre **Précision** et **Rappel**.

| Mesure      | Formule                          | Intuition |
|-------------|----------------------------------|-----------|
| Précision   | TP / (TP + FP)                   | Parmi les **positifs prédits**, combien sont corrects ? |
| Rappel      | TP / (TP + FN)                   | Parmi les **vrais positifs**, combien sont trouvés ? |
| F1-score    | 2 × (Précision × Rappel) / (Précision + Rappel) | Équilibre entre précision et rappel (moyenne harmonique). |

### 🧭 Quand l’utiliser

- **Données déséquilibrées** : l’accuracy peut être trompeuse.
- **Coûts FP/FN équilibrés** : F1 traite la précision et le rappel de façon symétrique.
- **Sinon, utiliser Fβ** pour privilégier l’un des deux :
  - **β > 1** → favorise le **rappel** (réduire les FN).
  - **β < 1** → favorise la **précision** (réduire les FP).

> Rappel : \(F_{\beta}\) généralise F1 (β=1).

### 🔑 Points clés

- **F1 ignore les vrais négatifs (TN)** : il ne “voit” que TP, FP, FN.
- **Dépend d’un seuil de décision** (comme précision/rappel).
- **Multiclasse** : agréger via `average=`  
  - `macro` : moyenne non pondérée des F1 par classe (chaque classe compte autant).  
  - `weighted` : moyenne pondérée par le support (peut masquer les classes rares).  
  - `micro` : cumule TP/FP/FN globalement (en mono-étiquette, **micro-F1 = accuracy**).

### Mémo Fβ (sans LaTeX)
Fβ = (1 + β²) · (Précision · Rappel) / (β² · Précision + Rappel)

### Code sklearn
```py
from sklearn.metrics import f1_score

# binaire
f1_binaire = f1_score(y_true, y_pred)

# multiclasse
f1_macro = f1_score(y_true, y_pred, average="macro")
f1_weighted = f1_score(y_true, y_pred, average="weighted")
f1_micro = f1_score(y_true, y_pred, average="micro")

```