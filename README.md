#  Projet Deep Reinforcement Learning – DQN pour Atari Breakout

##  Description

Ce package contient une implémentation complète d’un agent d’apprentissage par renforcement basé sur **Deep Q-Network (DQN)** appliqué à l’environnement Atari Breakout.

L’objectif est d’entraîner un agent capable d’apprendre à jouer uniquement à partir des pixels de l’écran, en maximisant le score obtenu via interaction avec l’environnement.

L’implémentation inclut plusieurs améliorations du DQN classique :

* Double DQN 
* Prioritized Experience Replay (échantillonnage efficace)
* Exploration ε-greedy avec décroissance
*  Monte-Carlo Dropout pour exploration par incertitude

---

##  Environnement d’exécution

* **Langage** : Python
* **Version recommandée** : Python 3.10 – 3.12
* **Plateforme testée** : Windows / Linux
* **Exécution** : CPU 

---

##  Structure du projet

```
/project_root
│
├── main.py                # Script principal d'entraînement
├── dqn_agent.py          # Implémentation du DQN / Double DQN
├── replay_buffer.py      # Replay buffer (standard ou priorisé)
├── model.py              # Architecture du réseau de neurones
├── utils.py              # Fonctions utilitaires
├── models/               # Modèles sauvegardés (.pth)
├── logs/                 # Logs d'entraînement
├── README.txt            # Instructions d'utilisation
```

---

## ▶️ Lancer l’entraînement

### Commande de base :

```bash
python main.py
```

---

## ⚙️ Paramètres principaux

Les paramètres peuvent être modifiés directement dans `main.py` :

| Paramètre       | Description                              |
| --------------- | ---------------------------------------- |
| `episodes`      | Nombre total d’épisodes d’entraînement   |
| `batch_size`    | Taille des mini-batchs                   |
| `gamma`         | Facteur de discount                      |
| `epsilon_start` | Valeur initiale de ε (exploration)       |
| `epsilon_end`   | Valeur minimale de ε                     |
| `epsilon_decay` | Vitesse de décroissance de ε             |
| `lr`            | Learning rate                            |
| `target_update` | Fréquence de mise à jour du réseau cible |
| `buffer_size`   | Taille du replay buffer                  |

---

## 💾 Sauvegarde du modèle

Les modèles sont automatiquement sauvegardés dans :

```
./models/
```

Exemple :

```
breakout_v5_v1.pth
```

---

## ▶️ Tester un modèle entraîné

Pour charger un modèle existant, modifier dans `main.py` :

```python
load_model = True
model_path = "./models/breakout_v5_v1.pth"
```

Puis exécuter :

```bash
python main.py
```

---


