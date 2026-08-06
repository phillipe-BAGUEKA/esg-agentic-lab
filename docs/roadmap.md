# Roadmap

Cette roadmap décrit une progression pédagogique et produit. Les technologies citées sont des pistes à étudier : leur adoption définitive dépendra des prototypes, des évaluations et des contraintes observées.

## V0 — Foundations

**Objectif :** établir un socle Python minimal, compréhensible, testable et documenté.

**Principales briques :** package en `src` layout, configuration de build, environnement local, test de fumée, charte, architecture provisoire et conventions de travail.

**Livrable de sortie :** un package installable en mode éditable, importable et validé par un premier test automatisé.

## V1 — Reliable RAG

**Objectif :** apprendre et valider une chaîne de recherche documentaire dont les réponses restent liées aux sources.

**Principales briques :** ingestion et segmentation contrôlées, représentation et indexation des contenus, retrieval, génération avec citations, jeux d'évaluation et mesure des erreurs. Les solutions d'embeddings, de stockage et d'orchestration seront comparées avant sélection.

**Livrable de sortie :** un prototype RAG évalué sur un corpus ESG limité, avec provenance des réponses et limites documentées.

## V2 — ESG Intelligence

**Objectif :** adapter les capacités documentaires aux besoins concrets de recherche ESG.

**Principales briques :** taxonomie métier, extraction d'indicateurs, comparaison d'émetteurs et de périodes, recherche de sources externes, détection et suivi de controverses, gestion de la temporalité.

**Livrable de sortie :** un parcours d'analyse ESG de bout en bout sur des cas représentatifs, revu par des utilisateurs métier.

## V3 — Agentic AI

**Objectif :** étudier une délégation contrôlée de tâches de recherche multiétapes.

**Principales briques :** outils spécialisés, planification, mémoire de travail, orchestration éventuelle de plusieurs agents, contrôles humains, budgets et garde-fous, traces d'exécution et évaluations agentiques.

**Livrable de sortie :** un assistant de recherche expérimental capable d'exécuter un protocole borné et auditable, avec validation humaine.

## V4 — Product and Production

**Objectif :** transformer les capacités validées en produit robuste et exploitable.

**Principales briques :** API et interface utilisateur — potentiellement FastAPI et Streamlit —, persistance, authentification, sécurité, observabilité, tests de charge, gestion des coûts et déploiement. PostgreSQL et pgvector feront partie des options évaluées.

**Livrable de sortie :** une version déployable, monitorée et documentée, assortie de critères d'exploitation et de procédures de retour arrière.
