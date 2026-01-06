import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import networkx as nx

# -----------------------------
# Paramètres généraux
# -----------------------------
np.random.seed(42)  # Pour des résultats reproductibles (à mentionner dans le rapport)

n_simulations = 10000

# -----------------------------
# Données du projet
# -----------------------------
tasks = {
    'A': {'a':5,  'm':7,  'b':10, 'urgency':1.2},
    'B': {'a':8,  'm':10, 'b':15, 'urgency':1.3},
    'C': {'a':6,  'm':8,  'b':12, 'urgency':1.4},
    'D': {'a':3,  'm':4,  'b':6,  'urgency':1.1},
    'E': {'a':12, 'm':15, 'b':20, 'urgency':1.5},
    'F': {'a':10, 'm':12, 'b':16, 'urgency':1.4},
    'G': {'a':5,  'm':6,  'b':8,  'urgency':1.2},
    'H': {'a':3,  'm':4,  'b':6,  'urgency':1.1},
    'I': {'a':4,  'm':5,  'b':7,  'urgency':1.2},
    'J': {'a':6,  'm':8,  'b':11, 'urgency':1.3},
    'K': {'a':2,  'm':3,  'b':4,  'urgency':1.1},
    'L': {'a':4,  'm':5,  'b':7,  'urgency':1.3},
    'M': {'a':3,  'm':4,  'b':6,  'urgency':1.1},
    'N': {'a':5,  'm':7,  'b':10, 'urgency':1.4},
    'O': {'a':4,  'm':5,  'b':7,  'urgency':1.0},
    'P': {'a':8,  'm':10, 'b':14, 'urgency':1.6},
    'Q': {'a':2,  'm':3,  'b':4,  'urgency':1.1},
    'R': {'a':1,  'm':2,  'b':3,  'urgency':1.0}
}

predecessors = {
    'A': [], 'B': ['A'], 'C': ['A'], 'D': ['A'],
    'E': ['B', 'C'], 'F': ['D'], 'G': ['E', 'F'],
    'H': ['G'], 'I': ['G'], 'J': ['G'],
    'K': ['H', 'I', 'J'], 'L': ['K'], 'M': ['L'],
    'N': ['M'], 'O': ['N'], 'P': ['N'],
    'Q': ['O', 'P'], 'R': ['Q']
}

# -----------------------------
# Construction du graphe et fonctions
# -----------------------------
G = nx.DiGraph([(pred, task) for task, preds in predecessors.items() for pred in preds])
topo_order = list(nx.topological_sort(G))

def simuler_duree(a, m, b):
    return np.random.triangular(a, m, b)

def calculer_duree_totale(durees):
    es = {task: 0 for task in tasks}
    for task in topo_order:
        if predecessors[task]:
            es[task] = max(es[pred] + durees[pred] for pred in predecessors[task])
    return es['R'] + durees['R']

# -----------------------------
# Analyse PERT déterministe
# -----------------------------
te = {task: (p['a'] + 4*p['m'] + p['b']) / 6 for task, p in tasks.items()}
sigma = {task: (p['b'] - p['a']) / 6 for task, p in tasks.items()}

total_te = calculer_duree_totale(te)
critical_tasks = ['A', 'B', 'E', 'G', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R']  # Pré-calculé

print(f"Durée totale attendue (déterministe) : {total_te:.2f} jours")
print(f"Chemin critique : {critical_tasks}")

# -----------------------------
# Simulation Monte Carlo - Scénario normal
# -----------------------------
resultats_normal = np.zeros(n_simulations)
durations_all = np.zeros((n_simulations, len(tasks) + 1))

for i in range(n_simulations):
    durees = {t: simuler_duree(p['a'], p['m'], p['b']) for t, p in tasks.items()}
    total = calculer_duree_totale(durees)
    resultats_normal[i] = total
    durations_all[i, :-1] = [durees[t] for t in tasks]
    durations_all[i, -1] = total

df_sim = pd.DataFrame(durations_all, columns=list(tasks.keys()) + ['total'])

mean_normal = np.mean(resultats_normal)
std_normal  = np.std(resultats_normal)

print(f"\nScénario normal :")
print(f"   Moyenne simulée : {mean_normal:.2f} jours")
print(f"   Écart-type      : {std_normal:.2f} jours")

# Probabilités demandées
print(f"   P(<100 jours)   : {np.mean(resultats_normal < 100)*100:.1f}%")
print(f"   P(<120 jours)   : {np.mean(resultats_normal < 120)*100:.1f}%")
print(f"   P(>130 jours)   : {np.mean(resultats_normal > 130)*100:.1f}%")
print(f"   P(110-125 jours): {np.mean((resultats_normal >= 110) & (resultats_normal <= 125))*100:.1f}%")

# Comparaison approximation PERT
mu_pert = sum(te[t] for t in critical_tasks)
std_pert = np.sqrt(sum(sigma[t]**2 for t in critical_tasks))
print(f"\nApproximation PERT : Moyenne = {mu_pert:.2f}, Écart-type = {std_pert:.2f}")

# Test de normalité
ks_stat, ks_p = stats.kstest(resultats_normal, 'norm', args=(mean_normal, std_normal))
print(f"Test KS : stat = {ks_stat:.4f}, p-value = {ks_p:.4f} → {'Similaire à normale' if ks_p > 0.05 else 'Diffère de normale'}")

# Tâches les plus influentes
corrs = df_sim.corr()['total'].drop('total').abs().sort_values(ascending=False)
print("\nTâches les plus critiques (influence sur variabilité) :")
print(corrs)

# -----------------------------
# Simulation - Scénario avec contrainte de ressources (urgence)
# -----------------------------
resultats_urgency = np.zeros(n_simulations)

for i in range(n_simulations):
    durees = {t: simuler_duree(p['a'], p['m'], p['b']) * p['urgency'] for t, p in tasks.items()}
    resultats_urgency[i] = calculer_duree_totale(durees)

mean_urgency = np.mean(resultats_urgency)
std_urgency  = np.std(resultats_urgency)

print(f"\nScénario avec contrainte de ressources (coefficients d'urgence) :")
print(f"   Moyenne simulée : {mean_urgency:.2f} jours")
print(f"   Écart-type      : {std_urgency:.2f} jours")
print(f"   P(<120 jours)   : {np.mean(resultats_urgency < 120)*100:.1f}%")

# -----------------------------
# Analyse de sensibilité (autres cas)
# -----------------------------
# 1. Réduction 10% sur chemin critique
tasks_red = {t: {**p, 'a':p['a']*0.9, 'm':p['m']*0.9, 'b':p['b']*0.9} for t, p in tasks.items()}
res_red = np.array([calculer_duree_totale({t: simuler_duree(p['a'], p['m'], p['b']) for t, p in tasks_red.items()}) 
                    for _ in range(n_simulations)])
print(f"\nSensibilité : -10% sur chemin critique → Moyenne = {np.mean(res_red):.2f} jours")

# 2. Retard 20% sur tâches les plus incertaines
most_uncertain = sorted(sigma, key=sigma.get, reverse=True)[:3]
tasks_del = tasks.copy()
for t in most_uncertain:
    tasks_del[t] = {**tasks_del[t], 'a':tasks_del[t]['a']*1.2, 'm':tasks_del[t]['m']*1.2, 'b':tasks_del[t]['b']*1.2}
res_del = np.array([calculer_duree_totale({t: simuler_duree(p['a'], p['m'], p['b']) for t, p in tasks_del.items()}) 
                    for _ in range(n_simulations)])
print(f"Sensibilité : +20% sur tâches les plus incertaines ({most_uncertain}) → Moyenne = {np.mean(res_del):.2f} jours")

# -----------------------------
# Visualisation : Comparaison normal vs urgence
# -----------------------------
plt.figure(figsize=(12, 7))

plt.hist(resultats_normal, bins=60, density=True, alpha=0.6, color='skyblue', label='Scénario normal')
plt.hist(resultats_urgency, bins=60, density=True, alpha=0.6, color='salmon',   label='Avec contrainte de ressources')

# Courbes normales
x1 = np.linspace(mean_normal - 4*std_normal, mean_normal + 4*std_normal, 100)
plt.plot(x1, stats.norm.pdf(x1, mean_normal, std_normal), 'blue', lw=2, label='Normale (normal)')

x2 = np.linspace(mean_urgency - 4*std_urgency, mean_urgency + 4*std_urgency, 100)
plt.plot(x2, stats.norm.pdf(x2, mean_urgency, std_urgency), 'red', lw=2, label='Normale (urgence)')

plt.title('Comparaison des durées simulées : Scénario normal vs Contrainte de ressources', fontsize=14)
plt.xlabel('Durée totale du projet (jours)')
plt.ylabel('Densité de probabilité')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('comparaison_normal_vs_urgence.png', dpi=300)
plt.show()

print("\nGraphique comparatif sauvegardé : 'comparaison_normal_vs_urgence.png'")