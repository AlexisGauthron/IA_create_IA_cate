# src/feature_engineering/llmfe/evolution_tracker.py
"""
Tracker pour suivre l'évolution des features générées par LLMFE.
Stocke l'historique et génère des visualisations/rapports.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union


@dataclass
class FeatureInfo:
    """Information sur une feature créée/modifiée."""

    name: str
    operation: str  # "created", "dropped", "transformed"
    source_columns: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SampleEvolution:
    """Évolution d'un sample (une itération LLM)."""

    sample_order: int
    score: Optional[float]
    score_delta: float  # Différence avec le meilleur score précédent
    is_best: bool

    # Temps
    sample_time: float
    evaluate_time: float
    timestamp: str

    # Code
    function_code: str

    # Features
    features_created: list[FeatureInfo] = field(default_factory=list)
    features_dropped: list[str] = field(default_factory=list)
    features_transformed: list[FeatureInfo] = field(default_factory=list)

    # Erreur éventuelle
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["features_created"] = [
            f.to_dict() if isinstance(f, FeatureInfo) else f for f in self.features_created
        ]
        d["features_transformed"] = [
            f.to_dict() if isinstance(f, FeatureInfo) else f for f in self.features_transformed
        ]
        return d


class EvolutionTracker:
    """
    Tracker pour suivre l'évolution des features LLMFE.

    Usage:
    ```python
    tracker = EvolutionTracker(output_dir="outputs/titanic/llmfe")

    # Après chaque évaluation
    tracker.record_sample(
        sample_order=1,
        score=0.82,
        function_code="def modify_features(df)...",
        sample_time=2.5,
        evaluate_time=0.1
    )

    # À la fin
    tracker.save()
    tracker.generate_report()
    ```
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        original_features: Optional[list[str]] = None,
        target_column: Optional[str] = None,
    ):
        """
        Args:
            output_dir: Dossier de sortie pour les fichiers
            original_features: Liste des features originales du dataset
            target_column: Nom de la colonne cible
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.original_features = original_features or []
        self.target_column = target_column

        # Historique
        self.samples: list[SampleEvolution] = []
        self.best_score: Optional[float] = None
        self.best_sample_order: Optional[int] = None

        # Métriques globales
        self.start_time = datetime.now()
        self.total_api_calls = 0
        self.total_errors = 0

        # Tracking des features au fil du temps
        self.feature_timeline: list[dict[str, Any]] = []

    def record_sample(
        self,
        sample_order: int,
        score: Optional[float],
        function_code: str,
        sample_time: float = 0.0,
        evaluate_time: float = 0.0,
        error: Optional[str] = None,
    ) -> SampleEvolution:
        """
        Enregistre un nouveau sample évalué.

        Args:
            sample_order: Numéro de l'itération
            score: Score obtenu (None si erreur)
            function_code: Code de la fonction générée
            sample_time: Temps de génération LLM
            evaluate_time: Temps d'évaluation
            error: Message d'erreur si échec

        Returns:
            SampleEvolution créé
        """
        # Calculer le delta
        score_delta = 0.0
        is_best = False

        if score is not None:
            if self.best_score is None:
                score_delta = score
                is_best = True
                self.best_score = score
                self.best_sample_order = sample_order
            else:
                score_delta = score - self.best_score
                if score > self.best_score:
                    is_best = True
                    self.best_score = score
                    self.best_sample_order = sample_order

        # Analyser le code pour extraire les features
        features_created, features_dropped, features_transformed = self._analyze_code(function_code)

        # Créer l'évolution
        evolution = SampleEvolution(
            sample_order=sample_order,
            score=score,
            score_delta=score_delta,
            is_best=is_best,
            sample_time=sample_time,
            evaluate_time=evaluate_time,
            timestamp=datetime.now().isoformat(),
            function_code=function_code,
            features_created=features_created,
            features_dropped=features_dropped,
            features_transformed=features_transformed,
            error=error,
        )

        self.samples.append(evolution)

        if error:
            self.total_errors += 1

        self.total_api_calls += 1

        # Mettre à jour la timeline des features
        self._update_feature_timeline(evolution)

        return evolution

    def _analyze_code(self, code: str) -> tuple:
        """
        Analyse le code pour extraire les opérations sur les features.

        Returns:
            (features_created, features_dropped, features_transformed)
        """
        features_created = []
        features_dropped = []
        features_transformed = []

        # Pattern pour détecter la création de nouvelles colonnes
        # df['NewCol'] = ... ou df_output['NewCol'] = ...
        create_pattern = r"df(?:_output)?\[(['\"])(\w+)\1\]\s*="
        for match in re.finditer(create_pattern, code):
            col_name = match.group(2)
            if col_name not in self.original_features:
                # Essayer de trouver les colonnes sources
                line_start = code.rfind("\n", 0, match.start()) + 1
                line_end = code.find("\n", match.end())
                line = code[line_start : line_end if line_end != -1 else len(code)]

                source_cols = self._extract_source_columns(line)

                features_created.append(
                    FeatureInfo(
                        name=col_name,
                        operation="created",
                        source_columns=source_cols,
                        description=self._infer_description(col_name, line),
                    )
                )

        # Pattern pour détecter les colonnes supprimées
        # .drop(['col1', 'col2'], ...) ou .drop(columns=['col1'])
        drop_patterns = [
            r"\.drop\(\s*\[([^\]]+)\]",
            r"\.drop\(\s*columns\s*=\s*\[([^\]]+)\]",
        ]
        for pattern in drop_patterns:
            for match in re.finditer(pattern, code):
                cols_str = match.group(1)
                cols = re.findall(r"['\"](\w+)['\"]", cols_str)
                features_dropped.extend(cols)

        # Pattern pour détecter les transformations
        # df['ExistingCol'] = df['ExistingCol'].transform(...)
        transform_pattern = (
            r"df(?:_output)?\[(['\"])(\w+)\1\]\s*=.*\2.*\.(fillna|replace|map|apply|astype)"
        )
        for match in re.finditer(transform_pattern, code):
            col_name = match.group(2)
            transform_type = match.group(3)
            if col_name in self.original_features:
                features_transformed.append(
                    FeatureInfo(
                        name=col_name,
                        operation="transformed",
                        description=f"{transform_type} applied",
                    )
                )

        return features_created, list(set(features_dropped)), features_transformed

    def _extract_source_columns(self, line: str) -> list[str]:
        """Extrait les colonnes sources utilisées dans une ligne de code."""
        source_cols = []
        # Pattern: df['col'] ou df_input['col']
        pattern = r"df(?:_input|_output)?\[(['\"])(\w+)\1\]"
        for match in re.finditer(pattern, line):
            col = match.group(2)
            if col in self.original_features:
                source_cols.append(col)
        return list(set(source_cols))

    def _infer_description(self, col_name: str, line: str) -> str:
        """Infère une description basée sur le nom et le code."""
        descriptions = []

        # Détection basée sur le nom
        name_lower = col_name.lower()
        if "family" in name_lower and "size" in name_lower:
            descriptions.append("Family size calculation")
        elif "title" in name_lower:
            descriptions.append("Title extraction from name")
        elif "age" in name_lower and "bin" in name_lower:
            descriptions.append("Age binning")
        elif "fare" in name_lower and "per" in name_lower:
            descriptions.append("Fare normalization")
        elif "alone" in name_lower:
            descriptions.append("Is alone indicator")

        # Détection basée sur les opérations
        if "+" in line:
            descriptions.append("Sum operation")
        if "/" in line:
            descriptions.append("Division operation")
        if "*" in line:
            descriptions.append("Multiplication operation")
        if ".str." in line:
            descriptions.append("String operation")
        if ".astype(int)" in line or ".astype(bool)" in line:
            descriptions.append("Type conversion")

        return " | ".join(descriptions) if descriptions else "Feature engineering"

    def _update_feature_timeline(self, evolution: SampleEvolution):
        """Met à jour la timeline des features."""
        entry = {
            "sample_order": evolution.sample_order,
            "score": evolution.score,
            "features_created": [f.name for f in evolution.features_created],
            "features_dropped": evolution.features_dropped,
            "features_transformed": [f.name for f in evolution.features_transformed],
            "total_features": len(self.original_features)
            + len(evolution.features_created)
            - len(evolution.features_dropped),
        }
        self.feature_timeline.append(entry)

    def get_summary(self) -> dict[str, Any]:
        """Retourne un résumé de l'évolution."""
        valid_samples = [s for s in self.samples if s.score is not None]
        scores = [s.score for s in valid_samples]

        # Features les plus créées
        all_created = {}
        for s in self.samples:
            for f in s.features_created:
                name = f.name if isinstance(f, FeatureInfo) else f.get("name", str(f))
                all_created[name] = all_created.get(name, 0) + 1

        # Features les plus supprimées
        all_dropped = {}
        for s in self.samples:
            for col in s.features_dropped:
                all_dropped[col] = all_dropped.get(col, 0) + 1

        return {
            "run_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
                "original_features": self.original_features,
                "target_column": self.target_column,
            },
            "statistics": {
                "total_samples": len(self.samples),
                "valid_samples": len(valid_samples),
                "failed_samples": self.total_errors,
                "success_rate": len(valid_samples) / len(self.samples) if self.samples else 0,
            },
            "scores": {
                "best_score": self.best_score,
                "best_sample_order": self.best_sample_order,
                "worst_score": min(scores) if scores else None,
                "mean_score": sum(scores) / len(scores) if scores else None,
                "score_improvement": (self.best_score - scores[0])
                if scores and len(scores) > 1 and self.best_score is not None
                else 0,
            },
            "feature_analysis": {
                "most_created_features": dict(
                    sorted(all_created.items(), key=lambda x: -x[1])[:10]
                ),
                "most_dropped_features": dict(
                    sorted(all_dropped.items(), key=lambda x: -x[1])[:10]
                ),
            },
            "timeline": self.feature_timeline,
        }

    def save(self, filename: str = "evolution_history.json") -> Path:
        """
        Sauvegarde l'historique complet en JSON.

        Args:
            filename: Nom du fichier

        Returns:
            Chemin du fichier créé
        """
        filepath = self.output_dir / filename

        data = {
            "summary": self.get_summary(),
            "samples": [s.to_dict() for s in self.samples],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"📊 Historique sauvegardé: {filepath}")
        return filepath

    def save_parquet(self, filename: str = "evolution_history.parquet") -> Path:
        """Sauvegarde en format Parquet pour analyse pandas."""
        try:
            import pandas as pd
        except ImportError:
            print("[WARNING] pandas non disponible pour export parquet")
            return None

        filepath = self.output_dir / filename

        records = []
        for s in self.samples:
            records.append(
                {
                    "sample_order": s.sample_order,
                    "score": s.score,
                    "score_delta": s.score_delta,
                    "is_best": s.is_best,
                    "sample_time": s.sample_time,
                    "evaluate_time": s.evaluate_time,
                    "n_features_created": len(s.features_created),
                    "n_features_dropped": len(s.features_dropped),
                    "features_created": ",".join(
                        [
                            f.name if isinstance(f, FeatureInfo) else str(f)
                            for f in s.features_created
                        ]
                    ),
                    "features_dropped": ",".join(s.features_dropped),
                    "has_error": s.error is not None,
                    "timestamp": s.timestamp,
                }
            )

        df = pd.DataFrame(records)
        df.to_parquet(filepath, index=False)

        print(f"📊 DataFrame sauvegardé: {filepath}")
        return filepath

    def generate_report(self, filename: str = "evolution_report.html") -> Path:
        """
        Génère un rapport HTML interactif.

        Args:
            filename: Nom du fichier HTML

        Returns:
            Chemin du fichier créé
        """
        filepath = self.output_dir / filename
        summary = self.get_summary()

        # Préparer les données pour les graphiques
        valid_samples = [s for s in self.samples if s.score is not None]
        scores_data = [(s.sample_order, s.score) for s in valid_samples]

        html_content = self._generate_html_report(summary, scores_data)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"📄 Rapport HTML généré: {filepath}")
        return filepath

    def _generate_html_report(self, summary: dict, scores_data: list[tuple]) -> str:
        """Génère le contenu HTML du rapport."""

        # Convertir les données pour Chart.js
        labels = [str(x[0]) for x in scores_data]
        values = [x[1] for x in scores_data]

        # Générer le tableau des samples
        samples_rows = ""
        for s in self.samples:
            status = "✅" if s.score else "❌"
            score_str = f"{s.score:.4f}" if s.score else "Error"
            delta_str = f"{s.score_delta:+.4f}" if s.score else "-"
            best_marker = "🏆" if s.is_best else ""

            created = (
                ", ".join(
                    [f.name if isinstance(f, FeatureInfo) else str(f) for f in s.features_created]
                )
                or "-"
            )
            dropped = ", ".join(s.features_dropped) or "-"

            samples_rows += f"""
            <tr class="{'best-row' if s.is_best else ''} {'error-row' if s.error else ''}">
                <td>{s.sample_order}</td>
                <td>{status} {score_str} {best_marker}</td>
                <td>{delta_str}</td>
                <td><code>{created}</code></td>
                <td><code>{dropped}</code></td>
                <td>{s.sample_time:.2f}s</td>
            </tr>
            """

        # Feature frequency
        feature_freq = summary["feature_analysis"]["most_created_features"]
        feature_freq_rows = ""
        for name, count in feature_freq.items():
            feature_freq_rows += f"<tr><td>{name}</td><td>{count}</td></tr>"

        return f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLMFE Evolution Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-color: #1a1a2e;
            --card-bg: #16213e;
            --text-color: #eee;
            --accent: #0f3460;
            --highlight: #e94560;
            --success: #4ade80;
            --warning: #fbbf24;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 2rem;
        }}

        .container {{ max-width: 1400px; margin: 0 auto; }}

        h1 {{
            font-size: 2.5rem;
            margin-bottom: 2rem;
            color: var(--highlight);
            text-align: center;
        }}

        h2 {{
            font-size: 1.5rem;
            margin: 1.5rem 0 1rem;
            color: var(--success);
            border-bottom: 2px solid var(--accent);
            padding-bottom: 0.5rem;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .stat-card {{
            background: var(--card-bg);
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            border: 1px solid var(--accent);
        }}

        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--highlight);
        }}

        .stat-label {{
            font-size: 0.9rem;
            color: #aaa;
            margin-top: 0.5rem;
        }}

        .chart-container {{
            background: var(--card-bg);
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            border: 1px solid var(--accent);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            background: var(--card-bg);
            border-radius: 10px;
            overflow: hidden;
        }}

        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--accent);
        }}

        th {{
            background: var(--accent);
            font-weight: 600;
        }}

        tr:hover {{ background: rgba(233, 69, 96, 0.1); }}

        .best-row {{ background: rgba(74, 222, 128, 0.1); }}
        .error-row {{ background: rgba(239, 68, 68, 0.1); }}

        code {{
            background: var(--accent);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.85rem;
        }}

        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }}

        @media (max-width: 768px) {{
            .two-col {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 LLMFE Evolution Report</h1>

        <h2>📊 Statistiques Globales</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{summary['statistics']['total_samples']}</div>
                <div class="stat-label">Total Samples</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary['statistics']['valid_samples']}</div>
                <div class="stat-label">Samples Valides</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{f"{summary['scores']['best_score']:.4f}" if summary['scores']['best_score'] else 'N/A'}</div>
                <div class="stat-label">Meilleur Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{f"{summary['scores']['score_improvement']:+.4f}" if summary['scores']['score_improvement'] else 'N/A'}</div>
                <div class="stat-label">Amélioration</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary['statistics']['success_rate']*100:.1f}%</div>
                <div class="stat-label">Taux de Succès</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary['run_info']['duration_seconds']:.1f}s</div>
                <div class="stat-label">Durée Totale</div>
            </div>
        </div>

        <h2>📈 Évolution des Scores</h2>
        <div class="chart-container">
            <canvas id="scoreChart"></canvas>
        </div>

        <div class="two-col">
            <div>
                <h2>🔬 Features les Plus Créées</h2>
                <table>
                    <thead>
                        <tr><th>Feature</th><th>Occurrences</th></tr>
                    </thead>
                    <tbody>
                        {feature_freq_rows if feature_freq_rows else '<tr><td colspan="2">Aucune feature créée</td></tr>'}
                    </tbody>
                </table>
            </div>
            <div>
                <h2>🗑️ Features les Plus Supprimées</h2>
                <table>
                    <thead>
                        <tr><th>Feature</th><th>Occurrences</th></tr>
                    </thead>
                    <tbody>
                        {"".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in summary['feature_analysis']['most_dropped_features'].items()]) or '<tr><td colspan="2">Aucune feature supprimée</td></tr>'}
                    </tbody>
                </table>
            </div>
        </div>

        <h2>📋 Détail des Itérations</h2>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Score</th>
                    <th>Delta</th>
                    <th>Features Créées</th>
                    <th>Features Supprimées</th>
                    <th>Temps</th>
                </tr>
            </thead>
            <tbody>
                {samples_rows}
            </tbody>
        </table>
    </div>

    <script>
        const ctx = document.getElementById('scoreChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {labels},
                datasets: [{{
                    label: 'Score',
                    data: {values},
                    borderColor: '#e94560',
                    backgroundColor: 'rgba(233, 69, 96, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 5,
                    pointHoverRadius: 8,
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Évolution du Score par Itération',
                        color: '#eee',
                        font: {{ size: 16 }}
                    }},
                    legend: {{ labels: {{ color: '#eee' }} }}
                }},
                scales: {{
                    x: {{
                        title: {{ display: true, text: 'Sample Order', color: '#aaa' }},
                        ticks: {{ color: '#aaa' }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }}
                    }},
                    y: {{
                        title: {{ display: true, text: 'Score', color: '#aaa' }},
                        ticks: {{ color: '#aaa' }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    def print_evolution_table(self):
        """Affiche un tableau récapitulatif dans le terminal."""
        print("\n" + "=" * 100)
        print("                           ÉVOLUTION DES FEATURES LLMFE")
        print("=" * 100)

        print(
            f"\n{'#':<4} {'Score':<12} {'Delta':<10} {'Features Créées':<30} {'Features Supprimées':<25}"
        )
        print("-" * 100)

        for s in self.samples:
            order = s.sample_order if s.sample_order else "-"
            score = f"{s.score:.4f}" if s.score else "ERROR"
            delta = f"{s.score_delta:+.4f}" if s.score else "-"
            best = " 🏆" if s.is_best else ""

            created = ", ".join(
                [f.name if isinstance(f, FeatureInfo) else str(f) for f in s.features_created[:3]]
            )
            if len(s.features_created) > 3:
                created += f" (+{len(s.features_created)-3})"
            created = created or "-"

            dropped = ", ".join(s.features_dropped[:3])
            if len(s.features_dropped) > 3:
                dropped += f" (+{len(s.features_dropped)-3})"
            dropped = dropped or "-"

            print(f"{order:<4} {score:<12}{best:<2} {delta:<10} {created:<30} {dropped:<25}")

        print("-" * 100)

        # Résumé
        summary = self.get_summary()
        print("\n📊 Résumé:")
        print(
            f"   • Meilleur score: {summary['scores']['best_score']:.4f} (sample #{summary['scores']['best_sample_order']})"
        )
        print(f"   • Amélioration: {summary['scores']['score_improvement']:+.4f}")
        print(f"   • Taux de succès: {summary['statistics']['success_rate']*100:.1f}%")

        if summary["feature_analysis"]["most_created_features"]:
            top_feature = list(summary["feature_analysis"]["most_created_features"].keys())[0]
            print(f"   • Feature la plus créée: {top_feature}")

        print("=" * 100 + "\n")
