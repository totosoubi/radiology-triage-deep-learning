# Rapport final - Systeme d'aide au tri radiologique

## Probleme

Le but du projet est de construire un prototype d'aide au tri radiologique a
partir de radiographies thoraciques. Le cas d'usage choisi est le suivant : a
partir d'une image, le systeme doit proposer des probabilites pour plusieurs
pathologies, signaler si l'image semble atypique, et utiliser un compte-rendu
quand une modalite texte est disponible.

Ce n'est donc pas seulement une classification simple. Le probleme est formule
en multi-label, car une meme radiographie peut avoir plusieurs anomalies en meme
temps. Les sorties du modele supervise sont donc 14 probabilites independantes,
une par pathologie.

## Donnees

Pour la partie principale, nous avons utilise ChestMNIST, qui fait partie de MedMNIST.
Le dataset est adapte au projet car il fournit des radiographies thoraciques et
14 labels de pathologies. Les splits officiels train, validation et test sont
gardes pour eviter de refaire une separation aleatoire difficile a comparer.

Pour la partie multimodale, nous avons ajoute OpenI/NLMCXR. Nous avons recupere les rapports
OpenI et un sous-ensemble d'images PNG de l'archive officielle. Le manifest local
contient 488 paires image + compte-rendu verifiees. Les labels OpenI ne sont pas
directement au meme format que ChestMNIST, donc nous avons construit une harmonisation
simple a partir des tags MeSH et du texte des sections `indication`, `findings`
et `impression`.

Les labels positifs les plus presents dans le sous-ensemble OpenI sont :
atelectasis 46, cardiomegaly 42, effusion 42, emphysema 27, nodule 25 et
infiltration 22. Ce sous-ensemble est suffisant pour valider la brique
multimodale, mais il reste trop petit pour tirer une conclusion clinique forte.

## Analyse exploratoire

La premiere observation sur ChestMNIST est le desequilibre important entre les
classes. Certaines pathologies sont relativement frequentes, alors que d'autres
sont presque absentes.

Labels les plus frequents dans le train :

| label | support | prevalence |
| --- | ---: | ---: |
| infiltration | 13914 | 0.1773 |
| effusion | 9261 | 0.1180 |
| atelectasis | 7996 | 0.1019 |
| nodule | 4375 | 0.0558 |
| mass | 3988 | 0.0508 |

Labels les plus rares dans le train :

| label | support | prevalence |
| --- | ---: | ---: |
| hernia | 144 | 0.0018 |
| pneumonia | 978 | 0.0125 |
| fibrosis | 1158 | 0.0148 |
| edema | 1690 | 0.0215 |
| emphysema | 1799 | 0.0229 |

Cette distribution justifie l'utilisation de metriques comme l'average precision
et l'AUROC, en plus du F1. Une accuracy classique serait peu informative ici.

Les figures generees pour cette partie sont :

- `artifacts/eda_label_stats.csv`
- `artifacts/eda_label_distribution.png`
- `artifacts/eda_cooccurrence_train.png`
- `artifacts/eda_train_samples.png`

## Preparation

Les images sont redimensionnees puis normalisees. Selon le modele, elles sont
chargees en un canal ou trois canaux. Le ResNet utilise trois canaux pour rester
compatible avec le transfer learning ImageNet, alors que le CNN simple et le ViT
leger peuvent travailler en niveaux de gris.

Les labels sont gardes au format multi-label binaire. La perte utilisee est
`BCEWithLogitsLoss`, ce qui permet de laisser le modele produire des logits et
d'appliquer ensuite une sigmoide par classe. Pour limiter l'effet du desequilibre
des classes, nous avons ajoute une option `--pos-weight`.

Nous avons aussi utilise une augmentation volontairement faible : petite rotation,
petite translation et changement d'echelle limite. L'idee est d'ameliorer un peu
la robustesse sans transformer les radiographies de facon trop artificielle.

Pour la reproductibilite, une seed est fixee, les splits officiels sont gardes,
et le meilleur modele est sauvegarde selon la validation.

## Modelisation supervisee

Trois architectures ont ete implementees, comme demande dans le sujet.

Le premier modele est un CNN simple entraine depuis zero. Il contient plusieurs
blocs convolution, batch normalization, ReLU et pooling, puis une tete lineaire.
Il sert surtout de baseline controlee.

Le deuxieme modele est un ResNet18 en transfer learning. L'interet est d'utiliser
des representations deja apprises sur ImageNet, puis de remplacer la derniere
couche pour produire les 14 sorties multi-label. Les connexions residuelles sont
un point important de cette architecture, car elles facilitent l'entrainement de
reseaux plus profonds.

Le troisieme modele est un petit Vision Transformer. Il decoupe l'image en patchs,
ajoute un embedding positionnel et utilise de l'attention. Ici le ViT reste assez
leger, car l'objectif est surtout de comparer la logique Transformer avec les
CNN, pas d'entrainer un tres grand modele.

Les entrainements utilisent AdamW, du weight decay, un scheduler cosine, un early
stopping simple, et une recherche de seuil sur validation pour ameliorer le F1
macro.

## Detection d'anomalies

Pour la detection d'anomalies, nous avons choisi un autoencodeur convolutionnel. Il est
entraine sur les images considerees comme normales, c'est-a-dire sans label
positif dans ChestMNIST. Le score d'anomalie correspond a l'erreur de
reconstruction MSE.

Le seuil est fixe au quantile 95 % des erreurs de reconstruction sur la
validation normale. Sur le test, 22433 images ont ete scorees. L'erreur moyenne
obtenue est 0.01730 et environ 6.6 % des images sont au-dessus du seuil.

Les artefacts principaux sont :

- `artifacts/ae_reconstructions.png`
- `artifacts/ae_error_histogram.png`
- `artifacts/ae_top_anomalies.png`
- `artifacts/ae_test_scores.csv`

Ce score doit rester interprete avec prudence. Une mauvaise reconstruction peut
venir d'une vraie anomalie, mais aussi d'une difference de contraste, de cadrage
ou d'un artefact d'image.

## Modelisation multimodale

Pour la partie multimodale, nous avons utilise le sous-ensemble OpenI prepare dans
`data/openi/manifest.csv`. Trois variantes ont ete comparees :

- image seule : petit encodeur CNN puis tete multi-label ;
- texte seul : vocabulaire local, embeddings et pooling masque ;
- fusion image + texte : concatenation des representations image et texte avant
  la tete de classification.

Les resultats moyens par classe sont les suivants :

| mode | mean_ap | mean_auc | mean_f1 |
| --- | ---: | ---: | ---: |
| image | 0.0902 | 0.5486 | 0.0815 |
| text | 0.1312 | 0.6024 | 0.1280 |
| fusion | 0.1854 | 0.6109 | 0.0654 |

La fusion obtient la meilleure average precision moyenne et le meilleur AUC
moyen, mais le F1 reste faible. Cela vient en partie du faible volume OpenI local
et du desequilibre des labels. Le resultat montre surtout que la brique
multimodale fonctionne et que le texte apporte une information utile.

## Evaluation

Les metriques suivies sont AUROC micro/macro, average precision micro/macro et
F1 micro/macro. L'average precision est importante dans ce projet, car les labels
positifs sont rares.

Synthese des derniers runs MLflow :

| experience | run | auc_micro | ap_micro | f1_macro |
| --- | --- | ---: | ---: | ---: |
| ChestMNIST | cnn renforce | 0.6335 | 0.0783 | 0.1179 |
| multimodal | fusion | 0.6986 | 0.1918 | 0.0654 |
| multimodal | text | 0.6846 | 0.1018 | 0.1280 |
| ChestMNIST | vit | 0.5670 | 0.0595 | 0.1014 |
| ChestMNIST | resnet18 | 0.5138 | 0.0497 | 0.1033 |
| multimodal | image | 0.4917 | 0.0438 | 0.0815 |

Nous avons relance le CNN sur un sous-ensemble plus large avec plus d'epochs.
Ce run ameliore l'AUROC micro et macro par rapport au premier essai. Pour rendre
l'evaluation plus solide, nous avons aussi calcule des intervalles de confiance
bootstrap sur 200 reechantillonnages :

| modele | auc_micro | IC 95 % auc_micro | ap_micro | IC 95 % ap_micro | f1_macro | IC 95 % f1_macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cnn | 0.6335 | [0.6274 ; 0.6393] | 0.0783 | [0.0755 ; 0.0808] | 0.1179 | [0.1150 ; 0.1205] |

Le ViT reste proche mais n'apporte pas de gain clair dans cette configuration.
Le ResNet18 n'est pas meilleur ici, probablement parce que les runs sont courts
et que le passage a des images medicales basse resolution limite l'avantage du
pre-entrainement ImageNet.

Quelques resultats par classe pour le CNN :

| label | support | AP | AUC | F1 |
| --- | ---: | ---: | ---: | ---: |
| atelectasis | 1279 | 0.1601 | 0.6470 | 0.2416 |
| effusion | 1417 | 0.2079 | 0.6894 | 0.2924 |
| infiltration | 2084 | 0.2510 | 0.6106 | 0.3235 |
| pneumothorax | 578 | 0.0583 | 0.5641 | 0.1044 |
| edema | 223 | 0.0495 | 0.7836 | 0.0766 |
| hernia | 19 | 0.0063 | 0.7393 | 0.0053 |

Les classes les plus rares restent les plus difficiles. Meme apres un run plus
large, `hernia` garde tres peu de positifs, donc le F1 reste peu stable.

Les nouveaux artefacts d'evaluation sont :

- `artifacts/scientific_summary.csv`
- `artifacts/per_class_thresholds.csv`
- `artifacts/curve_pr_micro_cnn.png`
- `artifacts/curve_roc_micro_cnn.png`

## Tracking MLflow

Toutes les experiences sont suivies avec MLflow dans `mlruns/`. Les scripts
enregistrent les hyperparametres, les metriques, les checkpoints, les figures et
les CSV de predictions. Nous avons aussi ajoute un export global :

```text
artifacts/mlflow_runs_summary.csv
```

Les experiences principales sont :

- `eda_chestmnist`
- `supervised_chestmnist`
- `anomaly_chestmnist_ae`
- `multimodal_openi_or_mimic`

Cela permet de verifier quel modele correspond aux artefacts utilises dans le
demonstrateur.

## Demonstrateur

Le demonstrateur est fait avec Streamlit. Il permet de charger une radiographie
et d'afficher :

- les predictions supervisees par pathologie ;
- le score d'anomalie de l'autoencodeur ;
- les predictions multimodales si un compte-rendu est saisi.

Les checkpoints charges sont :

- `artifacts/best_supervised.pt`
- `artifacts/best_ae.pt`
- `artifacts/best_multimodal_fusion.pt`

Commande de lancement :

```bash
streamlit run app/streamlit_app.py
```

## Analyse critique

Le projet respecte les differentes briques demandees, mais plusieurs limites
restent importantes.

D'abord, ChestMNIST est pratique pour experimenter, mais ce n'est pas le meme
niveau de complexite que des radiographies hospitalieres haute resolution. Les
performances obtenues ne doivent donc pas etre generalisees trop vite.

Ensuite, les classes sont tres desequilibrees. Certaines pathologies ont tres
peu d'exemples positifs, ce qui rend les metriques instables. Il faudrait des
runs plus longs, plus de donnees, et probablement des seuils par classe.

Pour l'autoencodeur, le score d'anomalie est utile pour signaler une image
atypique, mais il n'a pas de signification clinique directe. Il mesure surtout
une difficulte de reconstruction.

Enfin, la multimodalite a ete testee sur un vrai sous-ensemble OpenI, mais ce
sous-ensemble reste limite. Pour une evaluation plus solide, il faudrait utiliser
l'archive OpenI complete ou MIMIC-CXR.

## Conclusion et perspectives

Le projet aboutit a un prototype complet : classification multi-label sur
ChestMNIST, comparaison de trois architectures, detection d'anomalies,
multimodalite image + texte avec OpenI, suivi MLflow et interface Streamlit.

Les prochaines ameliorations seraient d'entrainer plus longtemps, d'utiliser
OpenI complet ou MIMIC-CXR, de calibrer les seuils par classe et d'ajouter une
analyse plus poussee des erreurs. En l'etat, le projet est surtout une preuve de
concept reproductible, pas un outil medical pret a etre utilise.

Configuration locale : macOS Darwin 25.5.0, Python 3.13.
