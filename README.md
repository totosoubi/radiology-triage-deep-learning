# Projet Deep Learning - tri radiologique

Ce projet a ete realise dans le cadre du cours de Deep Learning. L'objectif est
de construire un petit systeme de tri radiologique autour de radiographies
thoraciques : prediction multi-label de pathologies, detection de cas atypiques,
ajout d'une modalite texte, suivi des experiences et interface de test.

Le projet est surtout pense comme un prototype reproductible. Les resultats ne
doivent pas etre interpretes comme des resultats cliniques.

**Auteurs : Thomas Soubirou-Pouey et Estelle Letourneur**

## Organisation du dossier

- `src/radiotriage/` : code Python commun, datasets, modeles et metriques.
- `scripts/` : scripts pour l'EDA, les entrainements, MLflow et le rapport.
- `app/streamlit_app.py` : demonstrateur Streamlit.
- `artifacts/` : checkpoints deployes, figures, predictions et exports MLflow.
- `mlruns/` : runs MLflow.
- `rapport.md` : rapport final genere a partir des sorties du projet.
- `data/` : donnees locales, non versionnees.

## Installation

```bash
python3 -m pip install -r requirements.txt
```

Les scripts ont ete testes avec Python 3.13 sur macOS.

## Donnees utilisees

Pour la partie image supervisee, nous utilisons ChestMNIST via `medmnist`. Le fichier
est telecharge automatiquement dans `data/` au premier lancement.

Pour la partie multimodale, nous utilisons OpenI/NLMCXR. Les rapports sont dans
`NLMCXR_reports.tgz` et les images dans `NLMCXR_png.tgz`. Le script
`scripts/prepare_openi.py` construit ensuite un fichier
`data/openi/manifest.csv` avec les colonnes :

```text
image_path,report,atelectasis,cardiomegaly,...,hernia
```

Preparation OpenI :

```bash
python3 scripts/prepare_openi.py --download-reports --download-images --extract
```

Dans notre cas, l'archive image complete etait longue a telecharger. Nous avons
travaille avec un sous-ensemble reel extrait localement : 488 paires image +
compte-rendu verifiees. Le script ignore les PNG incomplets.

## Lancer les experiences

EDA ChestMNIST :

```bash
python3 scripts/run_eda.py
```

Classification supervisee :

```bash
python3 scripts/train_supervised.py --model cnn --epochs 8 --batch-size 64 --subset-size 30000 --image-size 64 --in-channels 1 --augment --pos-weight --patience 4
python3 scripts/train_supervised.py --model cnn --epochs 3 --batch-size 64 --subset-size 2000 --image-size 64 --in-channels 1 --augment --pos-weight
python3 scripts/train_supervised.py --model resnet18 --epochs 2 --batch-size 16 --subset-size 512 --image-size 64 --in-channels 3 --augment --pos-weight
python3 scripts/train_supervised.py --model vit --epochs 2 --batch-size 32 --subset-size 1000 --image-size 64 --in-channels 1 --augment --pos-weight
```

Detection d'anomalies par autoencodeur :

```bash
python3 scripts/train_anomaly.py --epochs 3 --batch-size 64 --subset-size 512 --image-size 64 --in-channels 1
```

Multimodal OpenI :

```bash
python3 scripts/train_multimodal.py --mode image --manifest data/openi/manifest.csv --epochs 5
python3 scripts/train_multimodal.py --mode text --manifest data/openi/manifest.csv --epochs 5
python3 scripts/train_multimodal.py --mode fusion --manifest data/openi/manifest.csv --epochs 5
```

La premiere commande correspond au run CNN utilise dans le rapport final. Les
autres commandes sont plus courtes et servent surtout de comparaison. Les options
`--subset-size` gardent des temps d'execution raisonnables sur machine locale ;
pour des resultats plus propres, il faut augmenter les epochs et reduire les
sous-echantillonnages quand la machine le permet.

## MLflow

Les experiences sont suivies avec MLflow dans `mlruns/`.

```bash
export MLFLOW_ALLOW_FILE_STORE=true
mlflow ui --backend-store-uri mlruns
```

Pour obtenir un CSV recapitulatif :

```bash
python3 scripts/export_mlflow_summary.py
```

Cette commande produit deux niveaux de preuve :

- `artifacts/mlflow_runs_summary.csv` : export complet des runs locaux ;
- `artifacts/mlflow_selected_runs.csv` et `artifacts/mlflow_selected_runs.png` :
  synthese des comparaisons, run retenu et checkpoints exposes dans Streamlit.

Pour recalculer les courbes PR/ROC, les seuils par classe et les intervalles de
confiance bootstrap :

```bash
python3 scripts/scientific_evaluation.py --models cnn --bootstrap 200
```

Pour generer l'analyse d'erreurs et les cartes Grad-CAM du CNN :

```bash
python3 scripts/interpretability_analysis.py --device cpu
```

Le rapport peut ensuite etre regenere avec :

```bash
python3 scripts/generate_report.py
```

## Demonstrateur

L'application Streamlit permet de tester le pipeline sur une image :

```bash
streamlit run app/streamlit_app.py
```

Elle charge les checkpoints presents dans `artifacts/` :

- `best_supervised.pt` pour les predictions image ;
- `best_ae.pt` pour le score d'anomalie ;
- `best_multimodal_fusion.pt` pour utiliser aussi un compte-rendu saisi.

Ces trois checkpoints sont versionnes dans le depot. Le demonstrateur fonctionne
donc apres installation des dependances, sans devoir relancer les entrainements.
Il faut charger une radiographie thoracique au format PNG, JPG ou JPEG. Les
donnees OpenI restent locales et ne sont pas necessaires pour tester une image
importee.

## Verification du rendu

| Livrable | Element a remettre | Emplacement ou commande |
| --- | --- | --- |
| Rapport final | Sections imposees et auteurs | `rapport.md` |
| Code | Supervise, AE, multimodal, MLflow | `scripts/`, `src/radiotriage/` |
| Demonstrateur | Application locale fonctionnelle | `streamlit run app/streamlit_app.py` |
| Preuve MLflow | Runs, selection et modeles deployes | `artifacts/mlflow_selected_runs.csv`, `artifacts/mlflow_selected_runs.png` |

## Points importants

- La classification est multi-label : une image peut avoir plusieurs labels.
- La perte utilisee est `BCEWithLogitsLoss`.
- Les sorties sont interpretees avec une sigmoide par classe.
- Les splits ChestMNIST officiels train/val/test sont conserves.
- Une seed est fixee par defaut a `42`.
- Le meilleur modele est sauvegarde selon la validation.
- Les resultats OpenI sont une preuve de concept sur un sous-ensemble reel, pas
  une validation clinique.
