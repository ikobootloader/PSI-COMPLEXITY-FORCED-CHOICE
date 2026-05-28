# Une variable de trop, une variable oubliée

*Sur la complexité conditionnelle des cibles dans les expériences psi — et sur ce qu'elle ne révolutionne pas*

---

## Le paradoxe de la Joconde

Si je vous demande de deviner, sans aucun indice, le tableau auquel je pense, vos chances sont infimes. Si j'ajoute « c'est un portrait de femme de la Renaissance », elles bondissent : vous proposez la Joconde, la Jeune Fille à la perle, la Dame à l'hermine. Vous êtes dans le bon voisinage.

Rien de mystérieux là-dedans. La difficulté de deviner une cible ne dépend pas seulement de la cible, mais de ce que le devineur sait déjà. Un objet « horriblement complexe » à décrire pixel par pixel devient simple dès qu'on en connaît la catégorie. En langage informationnel : sa **complexité conditionnelle** — sa complexité *relative à un savoir préalable* — s'effondre.

Cette idée, banale en théorie de l'information, a une conséquence intéressante pour la recherche sur les phénomènes psi (télépathie, clairvoyance, vision à distance). Mais avant d'en faire une « révolution », il faut être honnête sur un point : le champ n'a pas ignoré la complexité des cibles. Il en a même fait sa théorie centrale. La vraie question est ailleurs.

## Ce que le champ contrôle déjà

On lit souvent que les expériences psi « ne contrôlent pas » la probabilité de tomber juste par hasard. C'est inexact pour les protocoles modernes.

Dans le Ganzfeld standard, le sujet en privation sensorielle légère (yeux couverts, bruit blanc) verbalise ses impressions, puis on lui présente **un ensemble de jugement de quatre images** : la vraie cible et trois leurres. Il les classe ; un « hit » est attribué quand la vraie cible reçoit la note la plus élevée. L'espérance par hasard est donc fixée à 25 %, et la consigne recommande de choisir des leurres aussi *dissemblables* que possible. Autrement dit, le problème « tout le monde devine de l'eau et du rouge » est partiellement neutralisé par construction : une description vague n'aide que si elle distingue la cible des trois leurres.

Mieux : le confondeur que l'on croit parfois « négliger » a été repéré dès les premiers grands jeux de données. En réanalysant l'autoganzfeld, Hyman a montré que **le taux de succès d'une cible augmentait avec sa fréquence d'apparition** dans l'expérience — autour de 25 % pour une cible vue une seule fois, et jusqu'à environ 52 % pour celles réutilisées six fois ou plus. C'est, presque mot pour mot, l'intuition « certaines cibles sont plus faciles ». Sauf qu'elle est connue, documentée, et qu'elle a servi d'argument *contre* les résultats positifs, pas en leur faveur.

*(Ces chiffres proviennent du débat historique sur l'autoganzfeld ; je les rapporte de mémoire des analyses publiées et il vaut mieux les revérifier sur les sources primaires avant de les citer dans un travail formel.)*

## La théorie dominante repose déjà sur une mesure de complexité

On présente souvent l'observation d'Edwin May — les cibles « à forte signification » ou « à fort affect » réussissent mieux — comme une énigme en attente d'explication. En réalité, May n'a pas laissé cette observation en suspens : il en a tiré une théorie. Pendant deux décennies à la tête du programme STARGATE, puis aux Laboratories for Fundamental Research, il a soutenu que le **gradient de l'entropie de Shannon** de la cible est le facteur déterminant du « transfert d'information », au point d'en faire l'une des conclusions centrales de son bilan de carrière (*Anomalous Cognition: Remote Viewing Research and Theory*, May & Marwaha, McFarland, 2014).

L'entropie de Shannon est précisément une mesure de complexité. Dire « la complexité de la cible compte » n'est donc pas une rupture : c'est la position dominante du principal théoricien du domaine.

Cela ne dévalue pas l'idée. Cela en déplace simplement le centre de gravité — et c'est là que se loge l'apport réel.

## L'apport réel : conditionnel, pas absolu

La mesure de May est **absolue** : l'entropie est une propriété de l'image elle-même, indépendante de qui la regarde. Or l'exemple de la Joconde suggère que la grandeur pertinente n'est pas la complexité de la cible *en soi*, mais sa complexité *pour ce sujet précis*.

Un monument parisien est une cible facile pour quelqu'un qui a grandi avec ses images, difficile pour qui ne l'a jamais vu. La même image a deux niveaux de difficulté selon le répertoire conceptuel du devineur. C'est le passage d'une complexité absolue à une **complexité conditionnelle** — au sens où la complexité de Kolmogorov d'un objet *sachant* une information de référence est inférieure à sa complexité non conditionnée.

Voilà la proposition défendable : non pas « le champ a oublié la complexité », mais « le champ a mesuré la complexité de la cible, et pas celle de la cible relativement au sujet ».

## La distinction qu'il ne faut surtout pas escamoter

Reste un piège conceptuel, et c'est le plus important. La complexité conditionnelle peut jouer deux rôles statistiques radicalement différents, et on ne peut pas se servir des deux selon ce qui arrange l'argument :

- **Comme confondeur du hasard.** Si des cibles « faciles » gonflent le taux de coïncidences comptées comme succès, alors la variable agit sur la *distribution nulle*. Mais dans un design en choix forcé correctement randomisé, ce confondeur est en grande partie absorbé : ce qui reste, c'est la **similarité sémantique entre la cible et ses leurres**, presque jamais quantifiée. Si les leurres sont exotiques et la cible prototypique, le classement est biaisé. C'est là que l'argument mord — et il vise la composition de l'ensemble de jugement, pas une absence totale de contrôle.

- **Comme modérateur d'un signal réel.** Si les cibles faciles « laissent mieux passer » un signal psi authentique, alors la variable module un *effet réel* — ce qui présuppose l'existence du psi. C'est une hypothèse testable, mais distincte de la précédente, et on ne peut l'invoquer pour expliquer à la fois les faux positifs *et* les vrais positifs.

Tant que cette distinction n'est pas tranchée dans le protocole — quel rôle prête-t-on à la variable, et quelle prédiction chiffrée en découle ? — l'hypothèse reste une intuition séduisante plutôt qu'une proposition scientifique.

## Une proposition concrète, et ses pièges

L'idée actionnable est de remplacer le jugement humain subjectif par une mesure objective de proximité sémantique. Les outils existent : des embeddings multimodaux (de type CLIP) permettent de représenter à la fois une description verbale et une image dans un même espace vectoriel, et d'y calculer des distances. On peut alors :

1. **Quantifier la difficulté intra-ensemble** : mesurer la similarité sémantique entre la cible et ses trois leurres, pour rendre comparables des essais aujourd'hui hétérogènes.
2. **Stratifier les cibles** par complexité, plutôt que de les mélanger.
3. **Profiler — prudemment — le répertoire du sujet**, pour estimer la complexité *conditionnelle* de chaque paire sujet-cible.
4. **Pondérer** les succès par cette difficulté : réussir sur une cible conditionnellement complexe « vaut » plus que réussir sur une cible prototypique.

Trois mises en garde, sans lesquelles la proposition se retourne contre elle-même :

- **Les embeddings ne sont pas un sol neutre.** Ils encodent exactement les prototypes culturels que l'on cherche à mesurer — biais datés, fortement anglocentrés. Une proximité « tour / phare » dans l'espace vectoriel reflète une cooccurrence de corpus, pas nécessairement la structure cognitive du sujet.
- **Le calcul de distance suppose toujours un ensemble candidat.** On retrouve donc, par une autre porte, la question de la sélection des leurres : elle n'est pas résolue, elle est déplacée.
- **Le profilage et la pondération multiplient les degrés de liberté analytiques.** C'est le paradoxe de la démarche : elle dénonce une variable non contrôlée, et sa remédiation ajoute la flexibilité la plus difficile à discipliner. Sans **préenregistrement** intégral du pipeline (mesure du répertoire, métrique de complexité, schéma de pondération, plan d'analyse) avant toute collecte, ce dispositif peut fabriquer de la significativité aussi facilement qu'il prétend en retirer.

## Ce que cela apporterait, dans les deux cas

L'intérêt de l'approche est d'être *indifférente au résultat*.

Si aucun effet psi ne survit à ces contrôles, on aura tout de même produit une métrique de complexité conceptuelle individuelle, un instrument réutilisable en psychologie cognitive, et une explication parcimonieuse d'un demi-siècle de résultats instables : non pas un signal qui s'efface mystérieusement, mais une difficulté moyenne mal mesurée.

Si un effet résiduel survit — minuscule mais stable — on disposera pour la première fois d'une estimation de son ampleur isolée du bruit de la complexité conditionnelle, avec des prédictions chiffrées et préenregistrées.

Dans les deux cas, on progresse. C'est, au fond, ce qu'on attend d'un protocole.

## Conclusion : un raffinement, pas une rupture

Il serait malhonnête de présenter cette idée comme la résolution d'un mystère que personne n'avait vu. La probabilité de base est déjà contrôlée par le choix forcé ; les confondeurs liés aux cibles sont connus et débattus depuis les années 1980 ; et la complexité de la cible est, sous la forme de l'entropie, la pierre angulaire de la théorie dominante.

Ce qui reste — et qui mérite d'être poursuivi — tient en trois gestes : passer de la complexité *absolue* à la complexité *conditionnelle*, remplacer le jugement subjectif par une mesure sémantique explicite, et préenregistrer le tout. Moins spectaculaire qu'une refondation du domaine. Mais plus exact, donc plus utile.

La bonne question n'est pas « avons-nous assez de données ? », ni même « le psi existe-t-il ? ». C'est : *mesurons-nous la bonne grandeur, et la mesurons-nous relativement au bon référentiel ?*

---

### Repères bibliographiques

Les références ci-dessous sont réelles et vérifiables ; les chiffres précis (taux de succès, fréquences) devraient être recontrôlés sur les articles originaux avant toute citation formelle.

- Edwin C. May & Sonali Bhatt Marwaha (dir.), *Anomalous Cognition: Remote Viewing Research and Theory*, McFarland, 2014 — recueil incluant les travaux de May et Spottiswoode sur l'entropie de Shannon comme propriété intrinsèque des cibles.
- Ray Hyman & Charles Honorton, « A Joint Communiqué: The Psi Ganzfeld Controversy », *Journal of Parapsychology*, 1986 — les recommandations méthodologiques communes ; point de départ du débat sur les contrôles.
- Daryl Bem & Charles Honorton, « Does Psi Exist? », *Psychological Bulletin*, 1994 — présentation de la méta-analyse autoganzfeld (espérance 25 %).
- Sur la notion de complexité conditionnelle : la complexité de Kolmogorov et la distance d'information (Li & Vitányi, *An Introduction to Kolmogorov Complexity and Its Applications*) fournissent le cadre formel — à mobiliser comme analogie, en restant prudent sur le caractère non calculable de ces grandeurs.
