# Comparaison multi-cartes : entraînement direct et reprises de trajectoires

Expérience terminée ; les résultats sont résumés dans
`results/multimap_summary.json`. La configuration utilisée est
`configs/multimap.toml`. Les deux variantes ne diffèrent que par l'option
`--trajectory-restart`.

## Protocole

- Entraînement simultané sur 8×8, 10×10 et 12×12.
- Canvas commun 12×12, cartes plus petites centrées, cases hors carte marquées
  comme murs.
- Même CNN, même récompense et mêmes graines 42, 43 et 44.
- 63 environnements : 21 affectés en permanence à chaque taille. Chaque pas
  collecte donc exactement 21 transitions par taille.
- 512 pas par collecte, 5 époques PPO, lot de 2 016 transitions.
- Budget demandé de 9 000 000, soit 9 031 680 transitions effectives et
  3 010 560 par taille.
- Validation toutes les 483 840 transitions, 15 parties par taille, horizon de
  10 000 actions et départ standard de longueur 3.

Le meilleur modèle est choisi sur la moyenne non pondérée des scores normalisés
`pommes / (taille² - 3)`. Il ne faut pas utiliser la moyenne brute de pommes pour
comparer les cartes.

## Curriculum indépendant

Chaque taille possède sa propre porte de promotion et ses propres états conservés.
Les paliers correspondent à 10 %, 25 %, 50 % et 75 % de la surface. Une promotion
demande 25 % de réussites sur les 100 derniers épisodes éligibles. Après promotion,
25 % des départs restent standards ; un réservoir vide revient au départ standard.

Aucun état ne passe d'une taille à une autre. Les reprises d'entraînement recréent
les réservoirs et les paliers depuis zéro.

## Exécution

Afficher les six commandes sans les lancer :

```powershell
.\scripts\train_multimap_matrix.ps1
```

Lancer la matrice complète :

```powershell
.\scripts\train_multimap_matrix.ps1 -Execute
```

Ou commencer par la paire de la graine 42 :

```powershell
.\scripts\train_multimap_matrix.ps1 -Seeds 42 -Execute
```

Le script refuse d'écraser un dossier existant et s'arrête à la première erreur.

## Résultats et évaluation finale

Chaque run contient les rapports d'entraînement, les scores par taille et les
fichiers `metrics/map_exposure.json` et `metrics/trajectory_restart_N.json`.

Pour une évaluation indépendante, répéter la commande suivante pour chaque taille
et chaque checkpoint sélectionné :

```powershell
.venv\Scripts\snake-evaluate.exe runs/multimap-trajectory-42/best_model/best_model.zip --map-size 8 --episodes 100 --seed 93000 --horizon 10000 --output reports/multimap-trajectory-42-map8.json
```

Le rapport générique ne lance pas d'évaluation multi-carte automatiquement ; les
tailles doivent être demandées explicitement avec `snake-evaluate`.
