# Snake : PPO face à un solveur hamiltonien

Apprendre à manger une pomme est simple ; continuer quand le corps remplit la
carte l'est moins. Ce projet compare un agent PPO à des
solveurs déterministes, dans un environnement Snake Gymnasium.

Le PPO reçoit une grille égocentrique encodée par un CNN et 33 caractéristiques
vectorielles. Un curriculum optionnel reprend ses propres trajectoires à des
longueurs croissantes. Un même modèle peut jouer sur plusieurs tailles de carte.

## Résultats

Sur les tailles apprises, moyennes de trois seeds d'entraînement et 100 parties
de test par seed et par taille, depuis un serpent de longueur 3 :

| Carte | PPO direct : pommes / victoires | PPO curriculum : pommes / victoires |
| --- | ---: | ---: |
| 8×8 | 57,18 / 82,3 % | 57,76 / 90,0 % |
| 10×10 | 83,77 / 59,0 % | 85,39 / 76,0 % |
| 12×12 | 101,69 / 30,3 % | 110,91 / 55,0 % |

Chaque entraînement consomme 9,03 millions de transitions, réparties également
entre les trois tailles. Les meilleurs checkpoints sont sélectionnés en validation,
puis testés avec d'autres seeds et un horizon de 10 000 actions.

**Enseignement :** le curriculum aide à terminer les cartes apprises, mais ne
garantit pas la généralisation : seulement 1 à 5,3 % de victoires sur 9×9 et
11×11. Ces tests changent aussi la parité et le padding ; ils ne permettent pas
d'isoler une seule cause. [Chiffres par seed](results/multimap_summary.json).

Le solveur hamiltonien strict fournit une référence sans apprentissage. Il exige
une carte compatible avec un cycle et un départ ordonné sur ce cycle ; l'horizon
peut limiter sa réussite. Le tableau ci-dessus compare les deux PPO, pas le solveur.

## Utilisation

Python, PyTorch, Stable-Baselines3, Gymnasium, NumPy, TensorBoard et pytest.

```powershell
uv sync --extra cpu --extra train --extra dev --extra play --locked
uv run snake-train --config configs/ppo_8.toml --run-name ppo8
uv run snake-play --model runs/ppo8/best_model/best_model.zip
uv run snake-benchmark --ppo-model runs/ppo8/best_model/best_model.zip --episodes 100 --output reports/ppo8-vs-cycle.json
```

Remplacer `cpu` par `gpu` pour l'installation CUDA configurée dans le projet.
Le benchmark utilise la configuration du modèle et les mêmes seeds de jeu pour
PPO et le solveur. `--policy hamiltonian` permet les raccourcis ; `--policy planner`
sélectionne le planificateur BFS.

Les courbes et le résumé sont générés automatiquement après l'entraînement :

```powershell
uv run snake-report runs/ppo8
uv run snake-evaluate runs/ppo8/best_model/best_model.zip --episodes 100 --seed 93000 --output reports/ppo8-test.json
```

Les checkpoints ne sont pas fournis dans le dépôt. `configs/smoke.toml` permet un
test court ; la configuration 8×8 prévoit 7,5 millions de transitions.

[Guide du code](docs/code_guide.md) · [Multi-cartes](docs/multimap.md) ·
[Curriculum](docs/trajectory_restart.md)
