from transformers import pipeline

clf = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
labels = ["Support Technique","Facturation","Ressources humaines","Logistique","Commercial"]
clf("Impossible de me connecter au serveur", candidate_labels=labels, multi_label=False)
