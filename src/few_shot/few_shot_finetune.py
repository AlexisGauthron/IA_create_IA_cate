from __future__ import annotations

import math
import os

from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import LabelAccuracyEvaluator
from torch.utils.data import DataLoader

models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "distilbert-base-uncased",
    "distiluse-base-multilingual-cased-v2",
]

# =============================================================================
# Classe FewShotFinetune
# =============================================================================
# Renommée de 'few_shot_finetune' vers 'FewShotFinetune'
# Raison: Les classes Python doivent être en PascalCase (pas snake_case)
# =============================================================================


class FewShotFinetune:
    """Classe pour le fine-tuning few-shot de modèles de sentence transformers."""

    def __init__(self, train_df, dev_df):
        self.train_df = train_df
        self.dev_df = dev_df

        # --- Encodage des labels (mapping string -> id) ---
        label_encoder = train_df["label"].astype("category")
        self.label2id = {cat: i for i, cat in enumerate(label_encoder.cat.categories)}
        self.id2label = {i: cat for cat, i in self.label2id.items()}

        print(f"[INFO] Nombre de classes : {len(self.label2id)}\n")

        # 1) Exemples d’ENTRAÎNEMENT
        # SoftmaxLoss attend des PAIRES de phrases -> on duplique le texte: [text, text]
        self.train_samples = [
            InputExample(
                texts=[row["text"], row["text"]],  # <-- 2 phrases !!
                label=self.label2id[row["label"]],
            )
            for _, row in train_df.iterrows()
        ]

        # 2) Exemples de DEV
        self.dev_samples = [
            InputExample(
                texts=[row["text"], row["text"]],  # <-- idem ici
                label=self.label2id[row["label"]],
            )
            for _, row in dev_df.iterrows()
        ]

    def finetune(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 16,
        num_epochs: int = 3,
        steps_evauation: int = 50,
    ):
        print(f"[INFO] Début fine-tuning modèle {model_name}...\n")
        model = SentenceTransformer(model_name)

        train_dataloader = DataLoader(self.train_samples, shuffle=True, batch_size=batch_size)
        dev_dataloader = DataLoader(self.dev_samples, shuffle=False, batch_size=batch_size)

        # 3) Définir la loss de classification (SoftmaxLoss = NLI-style, 2 phrases)
        num_labels = len(self.label2id)
        train_loss = losses.SoftmaxLoss(
            model=model,
            sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
            num_labels=num_labels,
        )

        # 4) Evaluator basé sur la même SoftmaxLoss
        dev_evaluator = LabelAccuracyEvaluator(
            dataloader=dev_dataloader,
            name="dev",
            softmax_model=train_loss,
        )

        # 5) Entraînement
        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)

        output_dir = (
            "Modeles/finetune/finetuned_" + model_name.replace("/", "_") + f"_{num_epochs}ep"
        )
        os.makedirs(output_dir, exist_ok=True)

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=dev_evaluator,
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            evaluation_steps=steps_evauation,
            output_path=output_dir,
        )

        # 👇 Évaluation finale explicite
        final_acc = dev_evaluator(model)  # retourne un float entre 0 et 1
        print(f"[RESULT] Accuracy finale sur dev : {final_acc}")

        print(f"Modèle sauvegardé dans {output_dir}")
        # return output_dir, self.label2id, self.id2label
        return output_dir
