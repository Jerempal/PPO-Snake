# Lire et utiliser le code

| Fichier | Rôle |
| --- | --- |
| `env.py` | Règles, récompense, observations et rendu Snake |
| `policy.py` | CNN de la grille, MLP du vecteur et concaténation |
| `config.py` | Paramètres TOML et contrôles de validité |
| `train.py` | Construction de PPO, collecte SB3, sauvegardes |
| `score_evaluation.py` | Validation et sélection de `best_model` |
| `training/` | Multi-map et curriculum optionnels |
| `evaluate.py` | Évaluation des checkpoints sur des parties complètes |
| `baselines.py`, `expert.py` | Cycles hamiltoniens et planificateur BFS |
| `benchmark.py` | PPO contre un solveur, depuis des départs standards |
| `report.py`, `run_report.py` | Résumé et courbes d'entraînement |

## Du jeu à une décision

`SnakeEnv.step(action)` applique gauche/tout droit/droite et produit l'observation
suivante. Pendant `model.learn()`, SB3 appelle la politique pour collecter un rollout,
puis optimise PPO sur plusieurs minibatches. Le code propre au projet ne réécrit
pas cette boucle.

L'observation comporte sept plans (tête, corps, queue, pomme, murs, ordre du corps,
occupation) et 33 valeurs de position, direction, danger, raycasts et occupation.
Le canvas est carré et fixe ; la carte jouable peut être plus petite ou rectangulaire.
`--map-size N` ou `--width N --height M` changent la carte sans changer le canvas.
Ne pas changer `--grid-size` à l'inférence d'un checkpoint existant.

Limite connue : une dimension impaire dans un canvas pair entraîne un padding
asymétrique dépendant de l'orientation. Il n'est pas corrigé silencieusement, afin
de ne pas modifier la représentation sous laquelle les résultats ont été mesurés.

## Fichiers d'un entraînement

- `resolved_config.json` : paramètres effectifs, architecture comprise ; utile au rechargement.
- `metadata.json` : état du run et dates de début/fin uniquement.
- `best_model/best_model.zip` : meilleur checkpoint de validation, pas forcément le dernier.
- `final_model.zip` et `checkpoints/` : état final et sauvegardes périodiques.
- `metrics/`, TensorBoard et `report/` : scores, logs PPO et courbes.

Un score compte les pommes ; il ne mesure pas la récompense cumulée. Le score
d'entraînement avec reprises longues n'est pas comparable directement au score
de test depuis longueur 3. L'explained variance décrit le critique, pas le taux
de victoire ; les losses PPO ne sont pas des mesures de performance du jeu.

## Rapports et comparaison

```powershell
uv run snake-report runs/RUN_A runs/RUN_B --output-dir reports/comparison
uv run snake-report runs/RUN_A --evaluate --episodes 100
uv run snake-benchmark --ppo-model runs/RUN_A/best_model/best_model.zip --episodes 100
```

Sans `--evaluate`, un rapport ne joue aucune nouvelle partie. Avec cette option,
il évalue best/final à nouveau, sans cache. Pour le multi-map, employer
`snake-evaluate --map-size N` pour chaque taille. Le benchmark ne joue qu'une
taille par appel ; sans override, il utilise la taille par défaut du run.

Les anciennes options `snake-report --curve`, `snake-train --report-curve` et
`snake-benchmark --resume/--force-agent` ont été retirées. Les anciens résultats
restent sur disque. Aucun commit, hash ou registre d'artefacts n'est enregistré.
