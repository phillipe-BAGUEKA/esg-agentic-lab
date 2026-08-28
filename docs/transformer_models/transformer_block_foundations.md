# Fondations d'un bloc Transformer causal

Ce prototype assemble une self-attention causale et un réseau feed-forward
position-wise dans un unique bloc Transformer **pré-normalisé**. Il reçoit et
retourne des représentations de shape `(batch_size, sequence_length, d_model)`.

## Architecture pré-normalisée

Le calcul suit exactement ces deux étages :

```text
attention_input  = attention_norm(hidden_states)
attention_output = attention(attention_input)
after_attention  = hidden_states + attention_output

feed_forward_input  = feed_forward_norm(after_attention)
feed_forward_output = feed_forward(feed_forward_input)
output              = after_attention + feed_forward_output
```

`attention_output` contient seulement le résultat contextualisé produit par la
self-attention. `after_attention` y ajoute les représentations reçues par le
bloc grâce à la première connexion résiduelle. Le `output` final ajoute ensuite
le résultat du feed-forward à `after_attention` grâce à la seconde connexion
résiduelle.

Les connexions résiduelles conservent un chemin direct pour l'information et
les gradients. Si un sous-module produit zéro, son étage restitue donc exactement
son entrée au lieu de la remplacer.

## LayerNorm, pre-norm et post-norm

`LayerNorm(d_model)` normalise séparément le vecteur de chaque position sur sa
dernière dimension, puis applique une échelle et un biais apprenables. Elle ne
mélange ni les exemples du batch ni les positions de la séquence.

Dans une architecture **pre-norm**, la normalisation précède chaque sous-module :

```text
x + sublayer(LayerNorm(x))
```

Dans une architecture **post-norm**, elle vient après l'addition résiduelle :

```text
LayerNorm(x + sublayer(x))
```

Ce prototype choisit pre-norm pour rendre visibles les deux entrées normalisées
et conserver un chemin résiduel direct. Ce choix est également couramment utile
pour stabiliser l'optimisation de réseaux profonds, même si ce prototype ne
contient encore qu'un seul bloc.

## Feed-forward position-wise

Le feed-forward applique le même petit MLP à chaque position :

```text
expanded  = input_projection(hidden_states)   # d_model -> d_ff
activated = GELU(expanded)
output    = output_projection(activated)      # d_ff -> d_model
```

La première projection élargit la représentation afin d'offrir davantage de
dimensions de calcul. `GELU` apporte la non-linéarité : sans elle, la composition
des deux projections linéaires resterait une seule transformation linéaire. La
seconde projection revient à `d_model` afin de rendre possible l'addition
résiduelle.

Les couches linéaires agissent uniquement sur la dernière dimension. Le MLP ne
mélange donc jamais les positions entre elles : seule la self-attention échange
de l'information le long de la séquence.

## Shapes complètes d'un exemple

Pour `batch_size=8`, `sequence_length=10`, `d_model=64`, `num_heads=8`, donc
`d_head=8`, et `d_ff=256` :

| Tenseur | Shape |
| --- | --- |
| `hidden_states` | `(8, 10, 64)` |
| `attention_input` | `(8, 10, 64)` |
| Q, K et V avant séparation | `(8, 10, 64)` chacun |
| Q, K et V après séparation | `(8, 8, 10, 8)` chacun |
| scores, scores masqués et poids | `(8, 8, 10, 10)` chacun |
| masque causal | `(10, 10)` |
| sorties des têtes | `(8, 8, 10, 8)` |
| sortie concaténée | `(8, 10, 64)` |
| `attention_output` | `(8, 10, 64)` |
| `after_attention` | `(8, 10, 64)` |
| `feed_forward_input` | `(8, 10, 64)` |
| `expanded` et `activated` | `(8, 10, 256)` chacun |
| `feed_forward_output` | `(8, 10, 64)` |
| `output` | `(8, 10, 64)` |

Le batch et la longueur de séquence ne changent pas. Seule la largeur interne
du MLP passe temporairement de `64` à `256`.

## Nombre de paramètres

Pour une couche linéaire, le nombre de paramètres est
`dimensions_de_sortie × dimensions_d'entrée + biais`.

Avec `d_model=8`, `num_heads=2` et `d_ff=32` :

```text
self-attention : 4 × (8 × 8 + 8)                 = 288
deux LayerNorm : 2 × (8 + 8)                     =  32
feed-forward   : (32 × 8 + 32) + (8 × 32 + 8)   = 552
bloc complet   : 288 + 32 + 552                  = 872
```

Avec l'exemple plus large `d_model=64`, `num_heads=8` et `d_ff=256` :

```text
self-attention : 4 × (64 × 64 + 64)                  = 16 640
deux LayerNorm : 2 × (64 + 64)                        =    256
feed-forward   : (256 × 64 + 256) + (64 × 256 + 64)  = 33 088
bloc complet   : 16 640 + 256 + 33 088                = 49 984
```

## Limites volontaires

Aucun dropout n'est ajouté dans cette première version afin de garder les
inspections et les vérifications de causalité déterministes. Le prototype se
limite à un bloc causal : il ne contient ni embeddings de tokens ou de position,
ni empilement de blocs, ni tête de langage, logits, loss, boucle d'entraînement,
génération, cross-attention, architecture encodeur-décodeur ou modèle GPT
complet.
