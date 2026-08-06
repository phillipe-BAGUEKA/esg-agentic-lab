# ESG Agentic Lab

ESG Agentic Lab est un projet de plateforme d'AI Research traçable destinée aux analystes ESG.

## Problème métier

L'analyse ESG mobilise des sources nombreuses, hétérogènes et évolutives. Leur collecte, leur lecture, leur rapprochement et la vérification des affirmations demandent un travail manuel important, tout en imposant de conserver un lien clair entre chaque conclusion et les preuves qui la soutiennent.

L'utilisateur principal du projet est l'analyste ESG.

## Proposition de valeur

Le projet vise à faciliter la recherche ESG en aidant l'analyste à explorer des documents et des sources externes, à retrouver les éléments pertinents et à produire des analyses traçables. L'humain reste responsable de l'interprétation et de la décision finale.

## Capacités cibles

À terme, le projet pourra couvrir :

- le traitement de documents financiers et ESG ;
- l'ingestion, la segmentation et l'indexation de rapports ;
- la recherche sémantique et un RAG évalué ;
- la recherche Web et l'analyse de controverses ;
- l'utilisation d'outils et l'orchestration d'agents ;
- des interfaces backend et frontend ;
- les tests, l'observabilité et le déploiement.

Ces capacités constituent des orientations de travail et ne sont pas encore implémentées.

## État actuel

Le projet est en phase **V0 — Foundations**. Le dépôt contient uniquement un package Python minimal, sa configuration, un test de fumée et la documentation initiale.

## Aperçu de la roadmap

La progression envisagée va des fondations Python vers un RAG fiable, puis vers l'intelligence ESG, les approches agentiques et, enfin, la mise en production. Les choix technologiques seront validés progressivement par l'apprentissage et l'évaluation. Voir [docs/roadmap.md](docs/roadmap.md) pour le détail.

## Installation locale

Depuis PowerShell, à la racine du dépôt :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

Python 3.11 ou une version ultérieure est requis.

## Avertissement

ESG Agentic Lab est un outil de recherche en cours de développement. Il ne fournit aucun conseil d'investissement et ses résultats devront être vérifiés par un utilisateur qualifié.
