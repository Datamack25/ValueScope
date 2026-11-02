# Base CSV de ValueScope

Ce dossier est rempli automatiquement par la plateforme (mode « En ligne ») ou par `python scripts/actualiser_base.py`.

| Fichier | Contenu |
|---|---|
| companies_market.csv | Dernier instantané des indicateurs de marché par société |
| companies_financials.csv | États financiers annuels (format long : ticker, poste, année, valeur) |
| countries_imf.csv | Séries macro FMI (ou Banque mondiale en repli), format long |
| countries_wb.csv | Séries Banque mondiale complémentaires, format long |
| commodities_prices.csv | Cours de clôture quotidiens des matières premières |
| analyses_log.csv | Journal des valorisations enregistrées |

Sur Streamlit Community Cloud (disque effacé au redémarrage, sources souvent bloquées) :
1. lancez `python scripts/actualiser_base.py` sur votre ordinateur ;
2. envoyez les CSV produits dans ce dossier de votre dépôt GitHub ;
3. l'application les utilise automatiquement quand une source en direct est indisponible.
