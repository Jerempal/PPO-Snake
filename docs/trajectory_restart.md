# Curriculum de reprises de trajectoires

Activation : `--trajectory-restart` ; désactivation : `--no-trajectory-restart`.
Les paramètres sont dans la section `[trajectory_restart]` du TOML.

L'environnement conserve un réservoir borné de vrais états atteints par l'agent,
à des paliers de longueur proportionnels à la surface : 10, 25, 50 et 75 % dans
la configuration multi-map. Une promotion demande 25 % de réussites sur les
100 derniers épisodes éligibles. Après promotion, 25 % des départs restent courts.
Un réservoir vide provoque un départ court, jamais un état inventé.

Chaque worker conserve ses propres états. En multi-map, chaque taille a une porte
de promotion indépendante. Les évaluations démarrent toujours depuis longueur 3.
Les réservoirs et paliers repartent à zéro lors d'une reprise d'entraînement.

Ce curriculum a amélioré le taux de victoire moyen en 12×12 de 30,3 à 55,0 %
sur le test multi-map (trois seeds, 100 parties par seed), sans gain uniforme sur
toutes les seeds et tailles. Les départs longs synthétiques (`start_states`) sont
un outil de test séparé, pas le mécanisme de ce curriculum.
