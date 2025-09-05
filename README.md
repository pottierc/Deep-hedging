# Deep Hedging (Multi-Asset)

> Implémentation d’une couverture *data-driven* pour une option sur **deux actifs corrélés** via des réseaux de neurones.  
> Contexte : cours *Deep Learning in Finance* (2024–25).

---

## Objectifs

- **Apprendre** une stratégie de couverture dynamique ($\delta^1_t, \delta^2_t$) pour une option dont le payoff dépend de deux sous-jacents.
- **Paramétrer** la stratégie de delta-hedging par des réseaux de neurones (deep hedging).
- **Minimiser** l’**erreur de couverture finale** sur des trajectoires simulées.
- **Évaluer** la qualité de la couverture (moyenne/écart-type de l’erreur, histogrammes, profils de deltas).

---

## Problème

- Deux actifs $S^1_t, S^2_t$ suivent un modèle **log-normal corrélé** (pas de transaction costs) :
  
$$
\begin{aligned}
\log S^1_{t_{j+1}} &= \log S^1_{t_j} + \mu_1 \Delta t + \sigma_1 \sqrt{\Delta t}\, G^1_j, \\
\log S^2_{t_{j+1}} &= \log S^2_{t_j} + \mu_2 \Delta t + \sigma_2 \sqrt{\Delta t}\, G^2_j,
\end{aligned}
\qquad \text{avec} \qquad
\begin{pmatrix} G^1_j \\ G^2_j \end{pmatrix}
\sim \mathcal{N}\left(0, \begin{bmatrix} 1 & \rho \\ \rho & 1 \end{bmatrix}\right).
$$

- **Produit** : call **ATM** sur le **produit** des deux actifs
  $g(S^1_T,S^2_T)=\left(\frac{S^1_T\,S^2_T}{S^1_0\,S^2_0}-1\right)^+ .$

- **Portefeuille auto-financé** (cash rémunéré à $r$) :
  $$V_{t_{j+1}} = V_{t_j} + r\left(V_{t_j}-\sum_{k=1}^2 \delta^k_{t_j} S^k_{t_j}\right)\Delta t + \sum_{k=1}^2 \delta^k_{t_j}\,(S^k_{t_{j+1}}-S^k_{t_j}).$$

- **Paramètres de base** (modifiables) :  
  $T=1$ an, $N=100$ pas $\Delta t=T/N$, $\mu_1=0.025$, $\mu_2=-0.01$, $\sigma_1=0.22$, $\sigma_2=0.30$, $\rho=0.5$, $r=0.04$.

---

## Approche Deep Hedging

- À chaque date $t_j$, on approxime les deltas par des **réseaux de neurones** :
  $$  \delta^1_{t_j}=h^1_j(S^1_{t_j},S^2_{t_j}), \quad \delta^2_{t_j}=h^2_j(S^1_{t_j},S^2_{t_j}).$$
  
  (Pratique : un petit MLP par date, ou un MLP partagé conditionné par $t_j$.)

- La **prime initiale** $\pi_\theta(S^1_0,S^2_0)$ est aussi **apprise** (scalaire ou mini-réseau).

- **Loss** (MSE sur mini-lots de trajectoires) :
  $$  \mathcal L = \mathbb E\big[(\text{hedging error})^2\big],\quad \text{où } \text{hedging error}=\pi_\theta + \text{P\&L de couverture} - g(S^1_T,S^2_T).$$

---

## Contenu du dépôt

- `DL_in_finance_project_2_description.pdf` - énoncé du projet.
- `Part_1_Deep_hedging.ipynb` — rappel mono-actif et benchmark.
- `Part_2_Deep_hedging_multi_asset.ipynb` — cas **multi-actif corrélé**, simulation, modèle, entraînement, évaluation.
---

## Installation

# Dépendances minimales
pip install numpy pandas scipy matplotlib torch torchvision tqdm jupyter
