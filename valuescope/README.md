# ValueScope — Analyse financière & valorisation d'entreprises

Plateforme Streamlit inspirée de la démarche d'**Aswath Damodaran** (NYU Stern, *Narrative & Numbers*) :
importez les états financiers d'une société française, américaine ou autre, obtenez un diagnostic complet,
des prévisions, une valorisation multi-méthodes et un verdict **sous-évaluée / correctement valorisée / surévaluée**.

## Fonctionnalités

| Étape | Contenu |
|---|---|
| **Import** | Excel, CSV, PDF, Word (.docx), image (OCR Tesseract). Libellés PCG / liasse / IFRS (FR) et US GAAP 10-K (EN) reconnus automatiquement, agrégats manquants reconstitués, tableau éditable, multi-fichiers fusionnés par année, modèle Excel vierge. |
| **Diagnostic** | ~30 ratios (croissance, marges, ROE, ROIC, DuPont, gearing, dette nette/EBITDA, couverture, liquidité, DSO/DIO/DPO, conversion cash, réinvestissement), Altman Z / Z'', Piotroski F-score, stade du cycle de vie, commentaires d'analyste automatiques. |
| **Prévisions** | Scénarios pessimiste / central / optimiste sur 5 ans + tendance log-linéaire. |
| **Coût du capital** | rf de la devise, ERP mature (4,20 % au 1er oct. 2026), prime de risque pays × λ, bêta bottom-up sectoriel ré-endetté (Hamada), notation synthétique par couverture des intérêts, WACC et WACC stable. |
| **Valorisation** | DCF FCFF 2 phases à la Damodaran (convergence de marge, ventes/capital, IS effectif → marginal, g ≤ rf, ROIC terminal, probabilité de défaillance), DCF actionnaire FCFE/DDM, revenu résiduel (Ohlson/EVA), multiples (secteur ou comparables), ANR, Monte Carlo 5 000 tirages, tables de sensibilité, DCF inversé (croissance implicite du cours). |
| **Verdict & rapports** | Pondération des méthodes (adaptée au cycle de vie), football field, récit « narrative », exports **PDF, Word, Excel** (10 onglets + graphiques) et **images PNG** (zip). |
| **🏢 Grandes entreprises** | ~90 leaders (CAC 40, États-Unis, Europe, Asie & reste du monde) + tout ticker Yahoo : tableau comparatif (PER, EV/EBITDA, marges, ROE, dette, rendement, performance), carte qualité/prix, fiche société (cours 5 ans, comptes publiés, ratios, lecture d'analyste), **valorisation express** et bouton « Analyse complète » vers le moteur. |
| **🌍 Analyse pays** | 52 pays : carte mondiale et classements, fiche pays (dette brute/nette, soldes budgétaire et primaire, recettes/dépenses, croissance, inflation, chômage, balance courante, réserves, charge d'intérêts), **soutenabilité de la dette** (r − g, solde primaire stabilisant, choc de taux), score de risque, comparaison multi-pays. Données FMI WEO (projections incluses) + Banque mondiale. |
| **🛢 Matières premières** | 22 matières premières (pétrole, gaz, carburants, or, argent, platine, palladium, cuivre, aluminium, lithium, uranium, terres rares, blé, maïs, soja, café, cacao, sucre, coton, bétail) + dollar : panorama des performances, fiche (usages, producteurs, facteurs de prix, tendance, replis), base 100, corrélations, ratios or/argent et cuivre/or, écart Brent − WTI. |

| **🤖 Machine learning** | Typologie des sociétés (K-means + ACP) et **vrais comparables** par plus proches voisins (applicables aux multiples de la valorisation), **détection d'anomalies** (Isolation Forest), **prévision des matières premières** à 4 semaines (Gradient Boosting / Random Forest / Ridge, validation glissante contre la marche aléatoire, intervalles quantiles, backtest), **typologie des pays**, **risque de dégradation financière** (logistique / GBM sur le panel de la base ou sur votre CSV étiqueté, AUC en validation croisée, probabilité injectable dans le DCF). |
| **💾 Base de données** | Base CSV (`data/store/`) alimentée automatiquement : comptes et multiples des sociétés, séries FMI/Banque mondiale, cours des matières premières, journal des analyses. Mode hors-ligne, actualisation en masse, export/import zip. |

Deux sociétés fictives sont fournies pour tester (`data/`). Les trois onglets « marchés » interrogent les sources **en direct** :
le serveur doit avoir accès à Internet (c'est le cas sur Streamlit Community Cloud). Les données sont mises en cache 1 h (marchés) à 12 h (macro).

## Installation locale

```bash
pip install -r requirements.txt
# OCR des images (optionnel) : installer Tesseract + langue française
#   macOS : brew install tesseract tesseract-lang
#   Ubuntu : sudo apt install tesseract-ocr tesseract-ocr-fra
streamlit run app.py
```

## Déploiement Streamlit Community Cloud

1. **Remplacez tous les fichiers** du dépôt par ceux de cette version (y compris `core/`, `scripts/`, `data/`). La version
   déployée s'affiche en bas de la barre latérale (« ValueScope v4.1 ») : si elle n'apparaît pas, l'ancienne version tourne encore.
2. **Dossier `.streamlit/`** : Streamlit Cloud ne lit `config.toml` qu'à la **racine du dépôt**. Si `app.py` est dans un
   sous-dossier (ex. `valuescope/app.py`), copiez aussi `.streamlit/config.toml` à la racine du dépôt (thème clair et accent doré).
3. **Redémarrez** l'application (Manage app → Reboot) pour installer `requirements.txt`.

### Pourquoi « 403 Forbidden » sur Streamlit Cloud ?

Le FMI et Yahoo Finance bloquent fréquemment les adresses IP des serveurs partagés (Streamlit Cloud, AWS, GCP). L'application bascule
alors automatiquement sur la Banque mondiale, la SEC, la FRED, puis la **base CSV**. Pour une couverture complète, remplissez la base
**depuis votre ordinateur** (où ces sources répondent) et envoyez-la sur GitHub :

```bash
pip install -r requirements.txt
python scripts/actualiser_base.py          # sociétés + pays + matières premières (≈ 5 à 10 min)
# puis ajoutez data/store/*.csv au dépôt GitHub
```

L'application en ligne utilisera ces CSV dès qu'une source en direct est indisponible (badge orange « Base CSV locale »).

## Structure

```
app.py                 Interface Streamlit (13 pages)
core/schema.py         Modèle de données + dictionnaire de libellés FR/EN + lecture des nombres (1 234,5 / (123))
core/parser.py         Import Excel/CSV/PDF/Word/image, reconstitution des agrégats
core/analysis.py       Ratios, DuPont, Altman, Piotroski, cycle de vie, commentaires
core/market.py         rf, ERP, CRP, notation synthétique, référentiel sectoriel
core/forecast.py       Scénarios et tendance
core/valuation.py      DCF FCFF, Monte Carlo, DCF inversé, FCFE/DDM, revenu résiduel, multiples, verdict
core/reports.py        Graphiques PNG, Excel, Word, PDF
core/companies.py      Univers des grandes entreprises, conversion des comptes Yahoo Finance
core/countries.py      FMI / Banque mondiale, dynamique de la dette, score de risque pays
core/commodities.py    Matières premières : cours, performances, signaux
ui_markets.py          Pages Grandes entreprises, Analyse pays, Matières premières
ui_kit.py              Composants d'interface partagés
ui_ml.py               Pages Machine learning et Base de données
core/ml.py             Modèles scikit-learn (clusters, kNN, Isolation Forest, prévision, classification)
core/store.py          Base CSV : lecture, écriture, export/import
data/store/            CSV de la base (vide au départ, se remplit à l'usage)
scripts/actualiser_base.py  Remplit la base CSV depuis votre ordinateur
assets/                Polices DejaVu (accents dans le PDF)
data/                  Exemples FR (CSV) et US (Excel)
```

## Sources de données et replis automatiques

| Besoin | Source principale | Replis |
|---|---|---|
| Multiples, cours des sociétés | Yahoo Finance (yfinance) | API « chart » Yahoo (cours, nom, devise) → base CSV |
| États financiers | Yahoo Finance (yfinance) | **SEC EDGAR** (comptes officiels XBRL, sociétés US) → base CSV |
| Macro pays | FMI DataMapper (WEO, projections) | **Banque mondiale** (indicateur par indicateur) → base CSV |
| Matières premières | Yahoo Finance (futures) | API chart Yahoo → **FRED** (Brent, WTI, gaz, carburants, prix mensuels FMI) → base CSV |

Si une page affiche « Les données n'ont pas pu être chargées », elle indique la **cause probable** (limitation HTTP 429,
blocage 403, délai dépassé, module manquant) et propose **Réessayer** et **Tester toutes les sources**.
Le diagnostic complet est aussi dans 💾 Base de données. Sur Streamlit Cloud, Yahoo Finance limite souvent les requêtes :
chargez une dizaine de sociétés à la fois, ou utilisez le mode hors-ligne avec une base CSV remplie en local.

## Points d'attention

- **Base CSV sur Streamlit Cloud** : le disque est effacé à chaque redémarrage. Exportez la base (zip) depuis 💾 Base de données
  et placez les CSV dans `data/store/` du dépôt pour qu'ils soient rechargés.
- **Machine learning** : les modèles s'entraînent sur la base CSV ; plus elle est riche (sociétés, années), plus ils sont fiables.
  La qualité hors échantillon (silhouette, AUC, gain vs marche aléatoire) est toujours affichée : ne vous fiez pas à un modèle qui ne bat pas la référence.

- Les **paramètres de marché** (taux sans risque hors USD, primes pays, référentiel sectoriel) sont des **valeurs indicatives** :
  mettez-les à jour depuis le site de Damodaran (onglet *Data*) avant toute utilisation professionnelle.
- Après un import **PDF ou image**, vérifiez toujours le tableau dans l'onglet Import (mises en page très variables).
- Le nombre d'actions est en **millions** ; les montants dans l'unité choisie (unités, milliers, millions, milliards).
- Banques / assurances : privilégier FCFE et revenu résiduel.
- Outil d'aide à l'analyse — ne constitue pas un conseil en investissement.
