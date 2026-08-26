# Fondations LSTM et GRU

Ce document accompagne les implémentations manuelles de `ManualLSTM` et
`ManualGRU`. Elles utilisent les conventions de paramètres et l'ordre des
portes de PyTorch afin que chaque calcul puisse être comparé directement aux
modules officiels.

## LSTM : deux états complémentaires

Le LSTM transporte deux états :

- `c_t`, l'état de cellule, fournit un chemin mémoire additif contrôlé par les
  portes d'oubli et d'entrée ;
- `h_t`, l'état caché exposé à la position courante, est une version filtrée de
  l'état de cellule.

Avec l'ordre PyTorch `i, f, g, o` :

```text
i_t = sigmoid(W_ii x_t + b_ii + W_hi h_(t-1) + b_hi)
f_t = sigmoid(W_if x_t + b_if + W_hf h_(t-1) + b_hf)
g_t = tanh(   W_ig x_t + b_ig + W_hg h_(t-1) + b_hg)
o_t = sigmoid(W_io x_t + b_io + W_ho h_(t-1) + b_ho)

c_t = f_t * c_(t-1) + i_t * g_t
h_t = o_t * tanh(c_t)
```

`i_t` décide quelle nouvelle information écrire, `f_t` quelle mémoire
conserver, `g_t` propose un contenu candidat et `o_t` contrôle la partie de la
mémoire exposée dans `h_t`.

## GRU : un seul état

Le GRU fusionne mémoire et sortie dans `h_t`. Son ordre PyTorch est `r, z, n` :

```text
r_t = sigmoid(W_ir x_t + b_ir + W_hr h_(t-1) + b_hr)
z_t = sigmoid(W_iz x_t + b_iz + W_hz h_(t-1) + b_hz)
n_t = tanh(W_in x_t + b_in + r_t * (W_hn h_(t-1) + b_hn))

h_t = (1 - z_t) * n_t + z_t * h_(t-1)
```

La porte de reset `r_t` agit exactement sur la projection récurrente du
candidat. La porte de mise à jour `z_t` mélange l'ancien état et le nouveau
candidat.

## Différence LSTM / GRU

Le LSTM utilise quatre blocs de paramètres et deux états. Il sépare explicitement
le chemin mémoire `c_t` de la sortie `h_t`. Le GRU utilise trois blocs de
paramètres et un seul état ; il est donc plus compact. Aucun des deux n'est
universellement meilleur : le choix dépend du problème et doit être évalué.

## Shapes de l'exemple pédagogique

Pour `batch=8`, `sequence_length=10`, `input_size=4` et `hidden_size=50` :

| Élément | LSTM | GRU |
| --- | --- | --- |
| entrée | `(8, 10, 4)` | `(8, 10, 4)` |
| `output` | `(8, 10, 50)` | `(8, 10, 50)` |
| `h_n` | `(1, 8, 50)` | `(1, 8, 50)` |
| `c_n` | `(1, 8, 50)` | — |
| `weight_ih` | `(200, 4)` | `(150, 4)` |
| `weight_hh` | `(200, 50)` | `(150, 50)` |
| chaque biais | `(200,)` | `(150,)` |

Pour une couche unidirectionnelle, `output[:, -1, :]` est égal à `h_n[0]`.

## Nombre de paramètres

Chaque bloc possède des poids entrée-état, des poids état-état et deux biais :

```text
LSTM = 4 * hidden_size * (input_size + hidden_size + 2)
     = 4 * 50 * (4 + 50 + 2)
     = 11 200

GRU  = 3 * hidden_size * (input_size + hidden_size + 2)
     = 3 * 50 * (4 + 50 + 2)
     = 8 400
```

## Gradients et paramètres appris

Les portes, candidats, `c_t` et `h_t` sont des activations intermédiaires : ils
sont recalculés pour chaque entrée. Les matrices `weight_ih`, `weight_hh` et les
biais sont les paramètres appris et persistants.

Pendant la BPTT, les gradients traversent les états intermédiaires vers les
positions anciennes. L'optimiseur ne modifie pas directement ces états : il
met à jour les poids et les biais à partir des gradients accumulés.

Le chemin additif de `c_t` offre au gradient une route plus directe que la
récurrence RNN simple et peut atténuer le vanishing gradient. Il ne garantit ni
un gradient toujours préservé, ni l'absence d'exploding gradient. Initialisation,
longueur de séquence, données et optimisation restent importantes.

## Références

- [PyTorch `LSTMCell`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html)
- [PyTorch `LSTM`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [PyTorch `GRUCell`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRUCell.html)
- [PyTorch `GRU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)
