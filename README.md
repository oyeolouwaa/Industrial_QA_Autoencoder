# Industrial Quality Control — Bottle Anomaly Detection

Pipeline de vision industrielle fondé sur l'apprentissage non supervisé pour la détection automatisée et localisée d'anomalies sur des objets manufacturés (dataset MVTec AD, catégorie *bottle*). Développé en PyTorch et piloté via une interface web interactive sous Streamlit.

## Fonctionnement technique

Dans les lignes de production à haute cadence, les défauts de fabrication (fissures, éclats, impuretés, déformations) sont par nature rares et imprévisibles, ce qui rend impossible l'entraînement de classifieurs supervisés classiques. 

Ce projet résout ce problème par une approche non supervisée :
1. **Apprentissage exclusif de la normalité :** L'auto-encodeur convolutif (`ConvAutoencoder`) est entraîné uniquement sur des images de pièces conformes (`train/good`). Il apprend à compresser les caractéristiques structurelles fondamentales de l'objet dans un espace latent réduit, puis à les reconstruire fidèlement.
2. **Détection par erreur de reconstruction :** Lorsqu'une pièce présentant une anomalie est soumise au modèle, ce dernier ne parvient pas à reconstruire la zone altérée, car il ne l'a jamais apprise lors de sa phase d'apprentissage. 
3. **Cartes de chaleur (*Heatmaps*) :** Une mesure d'erreur quadratique moyenne (MSE) pixel par pixel est calculée entre l'image d'origine et l'image reconstruite. Un curseur de tolérance ajustable filtre le bruit de fond, et une palette thermique OpenCV superpose la zone défectueuse directement sur l'image d'origine.

## Structure du dépôt

```text
anomaly_detector/
├── data/
│   └── train/good/          # Images de référence saines (MVTec AD)
├── src/
│   ├── model.py             # Architecture PyTorch de l'auto-encodeur (encodeur/décodeur convolutif)
│   ├── train.py             # Pipeline d'entraînement, gestion du DataLoader et rétropropagation MSE
│   └── app.py               # Application Streamlit, calcul des résidus et overlay de la heatmap
├── autoencoder_anomaly.pth  # Poids du modèle entraîné
└── requirements.txt         # Dépendances du projet