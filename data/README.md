# Donnees attendues

## ChestMNIST

`chestmnist.npz` est telecharge automatiquement par `medmnist` dans ce dossier.
Le fichier est ignore par Git pour eviter de versionner les donnees.

## Manifest multimodal OpenI ou MIMIC-CXR

Preparation automatique OpenI :

```bash
python3 scripts/prepare_openi.py --download-reports --download-images --extract
```

Les archives officielles utilisees par le script sont :

- `https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz`
- `https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz`

Pour satisfaire completement la composante image + texte, ajouter un fichier :

```text
data/openi/manifest.csv
```

Colonnes obligatoires :

```text
image_path,report,atelectasis,cardiomegaly,effusion,infiltration,mass,nodule,pneumonia,pneumothorax,consolidation,edema,emphysema,fibrosis,pleural_thickening,hernia
```

`image_path` peut etre absolu ou relatif au dossier contenant le manifest.
`report` contient le compte-rendu radiologique. Les 14 labels sont binaires.

Exemple minimal :

```csv
image_path,report,atelectasis,cardiomegaly,effusion,infiltration,mass,nodule,pneumonia,pneumothorax,consolidation,edema,emphysema,fibrosis,pleural_thickening,hernia
images/CXR1.png,"No acute cardiopulmonary abnormality.",0,0,0,0,0,0,0,0,0,0,0,0,0,0
```

Commandes a relancer avec donnees reelles :

```bash
python3 scripts/train_multimodal.py --mode image --manifest data/openi/manifest.csv --epochs 10
python3 scripts/train_multimodal.py --mode text --manifest data/openi/manifest.csv --epochs 10
python3 scripts/train_multimodal.py --mode fusion --manifest data/openi/manifest.csv --epochs 10
python3 scripts/export_mlflow_summary.py
python3 scripts/generate_report.py
```
