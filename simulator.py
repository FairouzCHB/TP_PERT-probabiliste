import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import networkx as nx

tasks = {
    'A': {'a':5, 'm':7, 'b':10, 'urgency':1.2},
    'B': {'a':8, 'm':10, 'b':15, 'urgency':1.3},
    'C': {'a':6, 'm':8, 'b':12, 'urgency':1.4},
    'D': {'a':3, 'm':4, 'b':6, 'urgency':1.1},
    'E': {'a':12, 'm':15, 'b':20, 'urgency':1.5},
    'F': {'a':10, 'm':12, 'b':16, 'urgency':1.4},
    'G': {'a':5, 'm':6, 'b':8, 'urgency':1.2},
    'H': {'a':3, 'm':4, 'b':6, 'urgency':1.1},
    'I': {'a':4, 'm':5, 'b':7, 'urgency':1.2},
    'J': {'a':6, 'm':8, 'b':11, 'urgency':1.3},
    'K': {'a':2, 'm':3, 'b':4, 'urgency':1.1},
    'L': {'a':4, 'm':5, 'b':7, 'urgency':1.3},
    'M': {'a':3, 'm':4, 'b':6, 'urgency':1.1},
    'N': {'a':5, 'm':7, 'b':10, 'urgency':1.4},
    'O': {'a':4, 'm':5, 'b':7, 'urgency':1.0},
    'P': {'a':8, 'm':10, 'b':14, 'urgency':1.6},
    'Q': {'a':2, 'm':3, 'b':4, 'urgency':1.1},
    'R': {'a':1, 'm':2, 'b':3, 'urgency':1.0}
}

predecessors = {
    'A': [],
    'B': ['A'],
    'C': ['A'],
    'D': ['A'],
    'E': ['B', 'C'],
    'F': ['D'],
    'G': ['E', 'F'],
    'H': ['G'],
    'I': ['G'],
    'J': ['G'],
    'K': ['H', 'I', 'J'],
    'L': ['K'],
    'M': ['L'],
    'N': ['M'],
    'O': ['N'],
    'P': ['N'],
    'Q': ['O', 'P'],
    'R': ['Q']
}

G = nx.DiGraph()
for task in tasks:
    G.add_node(task)
for task, preds in predecessors.items():
    for pred in preds:
        G.add_edge(pred, task)

topo_order = list(nx.topological_sort(G))

def simuler_duree(a, m, b):
    return np.random.triangular(a, m, b)

def calculer_duree_totale(durees):
    es = {task: 0 for task in durees}
    for task in topo_order:
        if predecessors[task]:
            es[task] = max(es[pred] + durees[pred] for pred in predecessors[task])
    ef_r = es['R'] + durees['R']
    return ef_r

te = {}
sigma = {}
for task, params in tasks.items():
    a, m, b = params['a'], params['m'], params['b']
    te[task] = (a + 4*m + b) / 6
    sigma[task] = (b - a) / 6

total_te = calculer_duree_totale(te)

es = {task: 0 for task in te}
ef = {}
for task in topo_order:
    if predecessors[task]:
        es[task] = max(es[pred] + te[pred] for pred in predecessors[task])
    ef[task] = es[task] + te[task]

lf = {task: ef['R'] for task in te}
ls = {}
for task in reversed(topo_order):
    ls[task] = lf[task] - te[task]
    for pred in predecessors[task]:
        lf[pred] = min(lf[pred], ls[task])

critical_tasks = [task for task in te if abs(ls[task] - es[task]) < 1e-6]

print("Durée totale attendue (déterministe):", round(total_te, 2))
print("Tâches du chemin critique:", critical_tasks)

n_simulations = 10000
resultats = np.zeros(n_simulations)
durations_data = np.zeros((n_simulations, len(tasks) + 1))
task_list = list(tasks.keys())

for i in range(n_simulations):
    durees = {task: simuler_duree(params['a'], params['m'], params['b']) for task, params in tasks.items()}
    total = calculer_duree_totale(durees)
    resultats[i] = total
    durations_data[i, :-1] = [durees[task] for task in task_list]
    durations_data[i, -1] = total

durations_df = pd.DataFrame(durations_data, columns=task_list + ['total'])

mean_sim = np.mean(resultats)
std_sim = np.std(resultats)
print(f"Moyenne simulée: {mean_sim:.2f} jours")
print(f"Écart-type simulé: {std_sim:.2f} jours")

p_less_100 = np.mean(resultats < 100) * 100
p_less_120 = np.mean(resultats < 120) * 100
p_more_130 = np.mean(resultats > 130) * 100
p_between_110_125 = np.mean((resultats >= 110) & (resultats <= 125)) * 100

print(f"P(<100 jours): {p_less_100:.2f}%")
print(f"P(<120 jours): {p_less_120:.2f}%")
print(f"P(>130 jours): {p_more_130:.2f}%")
print(f"P(110 <= X <= 125): {p_between_110_125:.2f}%")

# Comparaison avec approximation normale
mu_pert = sum(te[task] for task in critical_tasks)
var_pert = sum(sigma[task]**2 for task in critical_tasks)
std_pert = np.sqrt(var_pert)
print(f"Approximation PERT - Moyenne: {mu_pert:.2f}, Écart-type: {std_pert:.2f}")

ks_stat, ks_p = stats.kstest(resultats, 'norm', args=(mean_sim, std_sim))
print(f"Test KS pour normalité: stat = {ks_stat:.4f}, p-value = {ks_p:.4f}")
if ks_p > 0.05:
    print("La distribution simulée est similaire à une normale (p > 0.05)")
else:
    print("La distribution simulée diffère de la normale (p <= 0.05)")

correlations = durations_df.corr()['total'].drop('total')
most_critical = correlations.abs().sort_values(ascending=False)
print("Tâches influençant le plus la variabilité (corrélation absolue):")
print(most_critical)

plt.figure(figsize=(10, 6))
plt.hist(resultats, bins=50, density=True, alpha=0.6, color='b', label='Simulée')
x = np.linspace(mean_sim - 4*std_sim, mean_sim + 4*std_sim, 100)
plt.plot(x, stats.norm.pdf(x, mean_sim, std_sim), 'r-', label='Normale approx.')
plt.title('Histogramme des durées simulées vs Approximation normale')
plt.xlabel('Durée totale (jours)')
plt.ylabel('Densité')
plt.legend()
plt.savefig('histogram_durees.png')
print("Histogramme sauvegardé sous 'histogram_durees.png'")


tasks_reduced = {k: v.copy() for k, v in tasks.items()}
for task in critical_tasks:
    tasks_reduced[task]['a'] *= 0.9
    tasks_reduced[task]['m'] *= 0.9
    tasks_reduced[task]['b'] *= 0.9

resultats_reduced = np.zeros(n_simulations)
for i in range(n_simulations):
    durees = {task: simuler_duree(params['a'], params['m'], params['b']) for task, params in tasks_reduced.items()}
    resultats_reduced[i] = calculer_duree_totale(durees)

mean_reduced = np.mean(resultats_reduced)
std_reduced = np.std(resultats_reduced)
print("\nSensibilité: Réduction de 10% sur chemin critique")
print(f"Nouvelle moyenne: {mean_reduced:.2f} jours")
print(f"Nouvel écart-type: {std_reduced:.2f} jours")

most_uncertain = sorted(sigma, key=sigma.get, reverse=True)[:3]
print("Tâches les plus incertaines:", most_uncertain)

tasks_delayed = {k: v.copy() for k, v in tasks.items()}
for task in most_uncertain:
    tasks_delayed[task]['a'] *= 1.2
    tasks_delayed[task]['m'] *= 1.2
    tasks_delayed[task]['b'] *= 1.2

resultats_delayed = np.zeros(n_simulations)
for i in range(n_simulations):
    durees = {task: simuler_duree(params['a'], params['m'], params['b']) for task, params in tasks_delayed.items()}
    resultats_delayed[i] = calculer_duree_totale(durees)

mean_delayed = np.mean(resultats_delayed)
std_delayed = np.std(resultats_delayed)
print("Sensibilité: Retard de 20% sur tâches les plus incertaines")
print(f"Nouvelle moyenne: {mean_delayed:.2f} jours")
print(f"Nouvel écart-type: {std_delayed:.2f} jours")

def simuler_duree_urgency(params):
    d = np.random.triangular(params['a'], params['m'], params['b'])
    return d * params['urgency']

resultats_urgency = np.zeros(n_simulations)
for i in range(n_simulations):
    durees = {task: simuler_duree_urgency(params) for task, params in tasks.items()}
    resultats_urgency[i] = calculer_duree_totale(durees)

mean_urgency = np.mean(resultats_urgency)
std_urgency = np.std(resultats_urgency)
print("Sensibilité: Avec contrainte de ressources (urgency)")
print(f"Nouvelle moyenne: {mean_urgency:.2f} jours")
print(f"Nouvel écart-type: {std_urgency:.2f} jours")