# Fondations de la self-attention causale

La self-attention transforme chaque position d'une séquence en combinant les
informations des positions qu'elle est autorisée à consulter. Dans ce prototype,
le masque causal interdit toute consultation du futur.

## Dimensions principales

- `d_model` est la largeur de représentation de chaque token.
- `num_heads` est le nombre de têtes calculées en parallèle.
- `d_head = d_model // num_heads` est la largeur réservée à chaque tête.

`d_model` doit donc être divisible par `num_heads`. Pour l'exemple pédagogique,
`d_model=8`, `num_heads=2` et `d_head=4`.

## Projections Q, K et V

Le même tenseur d'entrée `H`, de shape `(B, T, d_model)`, produit trois vues
apprenables :

```text
Q = H W_Q
K = H W_K
V = H W_V
```

Avant séparation, `Q` a la shape `(B, T, d_model)`. Il est ensuite réorganisé
en `(B, num_heads, T, d_head)`. `K` et `V` suivent la même transformation.
Cette opération partage les dimensions de représentation entre plusieurs têtes ;
elle ne crée pas de nouveaux tokens.

## Scores et transposition de K

Dans chaque tête, chaque query de shape `(d_head,)` est comparée à toutes les
keys. `K.transpose(-2, -1)` transforme les deux dernières dimensions de
`(T, d_head)` en `(d_head, T)`, ce qui rend possible le produit matriciel :

```text
Q @ K.transpose(-2, -1)
```

La shape résultante est `(B, num_heads, T, T)`. Les deux axes `T` représentent
respectivement la position query et la position key. Les scores sont divisés
par `sqrt(d_head)` avant le softmax pour limiter leur amplitude.

## Masque causal et poids

Le masque causal est une matrice triangulaire inférieure : une query à la
position `t` peut consulter les keys `0..t`, mais pas `t+1..T-1`. Les scores
interdits sont remplacés par `-inf`. Après le softmax sur l'axe des keys :

- les poids autorisés somment à `1` ;
- les poids vers le futur valent exactement `0`.

Chaque tête calcule ensuite une somme pondérée des values :

```text
head_output = attention_weights @ V
```

Sa shape est `(B, num_heads, T, d_head)`.

## Concaténation et projection de sortie

Les têtes sont remises côte à côte pour reconstruire
`(B, T, num_heads * d_head)`, soit `(B, T, d_model)`. Cette concaténation garde
les résultats de chaque tête dans des sous-espaces distincts.

`output_projection` est une dernière transformation linéaire apprenable de
`d_model` vers `d_model`. Elle mélange les informations produites par les têtes
et restitue l'interface attendue par la suite éventuelle du modèle.

Ce prototype s'arrête ici. Il n'ajoute ni normalisation, ni connexion résiduelle,
ni MLP, ni bloc Transformer complet, ni tête de langage.
