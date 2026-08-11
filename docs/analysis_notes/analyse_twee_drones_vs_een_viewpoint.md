# Analyse: meerwaarde van twee drones / twee viewpoints ten opzichte van een viewpoint

## Doel van deze notitie

Deze notitie bundelt wat de bestaande projectresultaten nu al zeggen over de kernvraag:

- hoe gebruik je een enkele drone met een enkel viewpoint het best;
- wat is de meerwaarde van twee drones met twee viewpoints;
- komt die winst vooral door extra dekking, door echte complementariteit, of door betere fusie van detecties;
- en hoe ontwerpen we een goed experimenteel model om dit netjes te onderbouwen.

De analyse hieronder is gebaseerd op bestaande resultaten in dit project en trekt daar een samenhangende conclusie uit.

## Korte hoofdconclusie

De centrale uitkomst is duidelijk: twee viewpoints zijn inhoudelijk waardevoller dan een viewpoint voor target-gedreven objectdetectie. De meerwaarde zit vooral in drie dingen:

1. een tweede drone kan een misser van de eerste drone opvangen;
2. een tweede viewpoint verhoogt de kwaliteit van de target-detectie, niet alleen de kans op een hit;
3. twee views leveren soms ook echt fuseerbare, elkaar bevestigende informatie op.

De grootste sprong zit tussen `1 -> 2` viewpoints. Een derde viewpoint helpt nog wel, maar levert veel minder extra op dan de tweede.

## Welke projectbestanden dit het sterkst onderbouwen

De belangrijkste onderliggende bestanden zijn:

- `m4_two_drone_operational_analysis/outputs/two_drone_operational_report.md`
- `m4_two_drone_operational_analysis/outputs/overall_one_vs_two_summary.csv`
- `m4_two_drone_operational_analysis/outputs/class_level_one_vs_two_summary.csv`
- `m4_marginal_viewpoint_value_analysis/outputs/marginal_value_report.md`
- `m4_cross_view_box_fusion_analysis/outputs/box_fusion_report.md`
- `m4_oracle_vs_box_fusion_comparison/outputs/oracle_vs_box_fusion_report.md`
- `m4_viewpoint_selection_analysis/outputs/robustness/ROBUST_VIEWPOINT_DIVERSITY_SUMMARY.md`
- `viewpoint_data_separated/single_vs_pair_comparison______pairtrained_vs_singleviewbaselines/outputs/single_vs_pair_summary.md`
- `outputs/m4_matched_control_experiment/reports/matched_control_report.md`
- `outputs/m4_matched_control_experiment/reports/master_results.csv`

Samen vertellen die bestanden eigenlijk hetzelfde verhaal vanuit drie hoeken:

- operationeel: wat gebeurt er als je tijdens inferentie een of twee drones hebt;
- trainingskant: wat gebeurt er als je modellen traint met een of twee viewpoints;
- fusiekant: wat gebeurt er als je detecties over meerdere views slim combineert.

## 1. Wat is het beste gebruik van een drone met een enkel viewpoint?

### Beste single-viewpoints

De sterkste losse viewpoints in de operationele analyse zijn:

- `ellow-radnear-az225`: target strict quality `0.9423`, target AP50-95 `0.8585`
- `elmid-radnear-az315`: target strict quality `0.9383`, target AP50-95 `0.9613`
- `elmid-radnear-az000`: target strict quality `0.9311`, target AP50-95 `0.9263`
- `elmid-radnear-az045`: target strict quality `0.9301`, target AP50-95 `0.9156`
- `elhigh-radnear-az000`: target strict quality `0.9291`, target AP50-95 `0.9378`

### Interpretatie

Daar zit een duidelijk patroon in:

- `near` radius komt vaak terug;
- `mid` of `low` elevation komt vaak terug;
- de beste viewpoints zijn niet willekeurig verdeeld over de 72 posities.

Dat suggereert dat een enkele drone het best wordt ingezet in een sterke, informatieve kijkhoek waarin:

- het target relatief groot in beeld komt;
- de kans op occlusie of ongunstige projectie kleiner is;
- de detector een stabiele confidence en IoU kan halen.

### Praktische boodschap voor 1 drone

Als je maar een drone hebt, dan wil je die niet "gemiddeld" laten kijken. Dan wil je juist een sterk viewpoint kiezen dat:

- dicht genoeg bij het object zit;
- niet te extreem hoog of te extreem ver staat;
- een hoek pakt die in de data consequent hoge target-kwaliteit geeft.

Een single-drone setup is dus vooral sterk als je vooraf al een goede positie kunt kiezen. De zwakte is dat je heel gevoelig blijft voor occlusie, een ongunstige objectorientatie of een toevallig moeilijk beeld.

## 2. Wat is de meerwaarde van twee drones ten opzichte van een drone?

## Operationele winst

De operationele vergelijking tussen `1 drone` en `2 drones` laat een duidelijke verbetering zien:

- expected target confidence: `0.9198 -> 0.9460`
- expected target strict quality: `0.8732 -> 0.9088`
- expected target AP50-95: `0.8516 -> 0.9277`
- binary target found rate: `0.9811 -> 0.9986`

De grootste headline-sprong is dus:

- `+0.0761` target AP50-95
- `+0.0357` target strict quality
- `+0.0175` target-found rate

Dit is substantieel. Het laat zien dat een tweede drone niet alleen een theoretisch voordeel heeft, maar in deze data ook praktisch betere target-detectie oplevert.

### Belangrijke nuance

De gewone gemiddelde beeldscore over geselecteerde beelden (`expected mean AP50-95`) blijft vrijwel gelijk (`0.7857 -> 0.7857`). Dat lijkt op het eerste gezicht alsof twee drones niet helpen, maar dat is hier misleidend.

De reden is dat de operationele vraag niet is: "is het gemiddelde losse beeld mooier?", maar:

- vind ik het target beter;
- heb ik een betere best beschikbare view;
- kan ik een misser van view 1 opvangen met view 2.

Daarom zijn `target AP50-95`, `strict quality` en `rescue rate` hier veel betere hoofdmetrics dan de gewone gemiddelde beeldscore.

## De tweede drone werkt vooral als rescue-view

De sterkste tweede-drones redden de situatie vaak volledig wanneer de eerste drone mist. In de huidige resultaten zie je bijvoorbeeld:

- `elmid-radmid-az000`: rescue rate given primary miss `1.0000`
- `elmid-radnear-az045`: rescue rate given primary miss `1.0000`
- `elmid-radfar-az090`: rescue rate given primary miss `1.0000`
- `elmid-radmid-az045`: rescue rate given primary miss `1.0000`
- `elhigh-radmid-az180`: rescue rate given primary miss `1.0000`

Dat is inhoudelijk belangrijk. De meerwaarde van twee drones zit dus niet alleen in "twee keer hetzelfde kijken", maar juist in:

- een drone als primaire view;
- een tweede drone als aanvullende reddingsview wanneer de eerste geen goede detectie geeft.

## Klassen die het meest winnen van twee drones

De meerwaarde van twee viewpoints is niet voor alle objectklassen even groot. De grootste winsten zitten bij:

- `barrel`: `+0.1094` target AP50-95, `+0.0806` strict quality
- `male`: `+0.0942` target AP50-95, `+0.0670` strict quality
- `suv`: `+0.0895` target AP50-95, `+0.0528` strict quality
- `tank`: `+0.0999` target AP50-95, `+0.0521` strict quality
- `whitevan`: `+0.0859` target AP50-95, `+0.0363` strict quality

Dat wijst erop dat meerdere viewpoints vooral nuttig zijn voor objecten die:

- compact of deels afgedekt zijn;
- op sommige hoeken makkelijk verward kunnen worden;
- of sterk afhankelijk zijn van pose en zichtbaarheid.

Voor zulke objecten is een tweede drone geen luxe, maar een echte robuustheidsversterker.

## 3. Waarom zijn twee viewpoints beter dan een viewpoint?

De projectresultaten laten zien dat daar drie verschillende mechanismen achter zitten.

### Mechanisme A: rescue-effect

De tweede drone ziet soms simpelweg wat de eerste mist. Dit is de meest directe vorm van meerwaarde.

### Mechanisme B: viewpoint-complementariteit

Sommige viewpoint-paren vullen elkaar aan, ook als beide individueel al redelijk sterk zijn. In de marginale analyse is complementarity gedefinieerd als:

`E[max(view_i, view_j)] - max(E[view_i], E[view_j])`

Voor ondersteunde paren zie je bijvoorbeeld:

- `ellow-radfar-az000 + ellow-radmid-az225`: strict-quality complementarity `+0.1209`
- `elhigh-radmid-az090 + ellow-radmid-az000`: `+0.0709`
- `elhigh-radfar-az000 + ellow-radfar-az135`: `+0.0705`
- `elhigh-radfar-az090 + ellow-radnear-az000`: `+0.0683`

Dat betekent: sommige paren zijn niet alleen goed omdat een van de twee sterk is, maar omdat de combinatie echt extra informatie toevoegt.

### Mechanisme C: fuseerbare evidentie

De box-fusion analyses laten zien dat meerdere views soms ook samen sterker worden dan de beste losse box:

- `2 views`, oracle/current method: `0.9088`
- `2 views`, support-weighted OR: `0.9212`
- `2 views`, noisy-OR + best IoU: `0.9515`

Dus zelfs een conservatieve, deployable fusiestrategie geeft nog:

- `+0.0124` boven de huidige best-view selectie bij twee views

Dat is inhoudelijk sterk bewijs dat meerdere drones niet alleen extra kans geven op een goede view, maar soms ook echt bevestigende informatie leveren die je kunt combineren.

## 4. Welke twee viewpoints combineren het best?

## Exacte topcombinaties

De sterkste exacte paren in de operationele analyse zijn:

- `elmid-radnear-az135 + elmid-radnear-az315`: strict quality `0.9659`, target AP50-95 `1.0000`
- `ellow-radnear-az225 + elmid-radfar-az000`: strict quality `0.9656`, target AP50-95 `0.9688`
- `ellow-radmid-az135 + elmid-radnear-az315`: strict quality `0.9633`, target AP50-95 `0.9914`
- `ellow-radmid-az045 + ellow-radfar-az315`: strict quality `0.9632`
- `ellow-radfar-az000 + elmid-radnear-az045`: strict quality `0.9631`

Maar hier hoort een belangrijke methodologische waarschuwing bij: exacte paren hebben vaak lage support. Voor een thesis-veilige conclusie moeten we daarom vooral naar robuuste relatiepatronen kijken.

## Robuuste relatiepatronen

De robustheidsanalyse over scenes geeft sterkere, generaliseerbare richtlijnen:

- beste `k=2` azimuth-relatie: `diagonal_135`, AP50-95 `0.9306`
- beste `k=2` afstandsrelatie: `near_far`, AP50-95 `0.9285`
- beste `k=2` elevatierelatie: `adjacent_elevation`, AP50-95 `0.9336`
- beste `k=2` mixed-diversity type: `elevation_only`, AP50-95 `0.9381`

Matched-scene verschillen laten ook zien dat:

- `distance+elevation` beter is dan `distance_only` met `+0.0335`
- `elevation+azimuth` beter is dan `distance_only`
- `adjacent_elevation` beter is dan `same_elevation` met `+0.0122`

### Inhoudelijke interpretatie

Twee drones moeten dus niet simpelweg een kopie van elkaar zijn. De data suggereert:

- kies liever verschillende hoogtes dan exact dezelfde hoogte;
- combineer liever nabij en verder weg dan twee identieke afstanden;
- een hoekverschil is nuttig, maar niet elk hoekverschil is even goed;
- pure afstandsvariatie zonder andere diversiteit is relatief zwak.

De slechtste ontwerpkeuze lijkt daarom: twee drones die vooral alleen maar "meer van hetzelfde" doen.

## 5. Wat zeggen de trainingsresultaten over 1 viewpoint versus 2 viewpoints?

De trainingskant bevestigt het beeld, maar met een belangrijke nuance.

## Single-view versus pair-view training

Uit de trainingsvergelijking:

- beste single-view trained model: `0.4164` mAP50-95
- beste pair-trained model: `0.4958` mAP50-95
- beste pair boven beste single: `+0.0794`
- paren die hun beste constituent single verslaan: `2380 / 2535` (`93.9%`)
- mean pair lift boven beste constituent single: `+0.0598`
- median pair lift: `+0.0360`

Dat is een sterk signaal dat twee viewpoints ook aan de trainingskant beter generaliseren dan een enkel viewpoint.

## Maar: hier zit een image-count confound in

Pairs zien gemiddeld ongeveer twee keer zoveel trainingsbeelden:

- mean single-view training images: `143.5`
- mean pair-view training images: `287.0`

Dus de ruwe pair-vs-single trainingswinst komt door een mix van:

- meer viewpoint-diversiteit;
- meer trainingsdata.

## Waarom matched controls belangrijk zijn

De matched-control experimenten trekken dat gedeeltelijk uit elkaar:

- full-M4 control matched to best single: `0.4803`, dat is `+0.0638` boven de best single source
- full-M4 control matched to best pair: `0.5196`, dat is `+0.0238` boven de best pair source

De interpretatie hiervan is belangrijk:

- niet alleen het aantal beelden telt;
- ook de breedte van de viewpoint-diversiteit in de training telt;
- een model dat evenveel beelden ziet, maar uit een rijkere viewpoint-verdeling, generaliseert beter dan een model dat vastzit aan een enkel viewpoint of een enkel viewpoint-paar.

De beste trainingsstrategie is dus waarschijnlijk niet: "train precies op een vast duo", maar eerder:

- train op een diverse multi-view verdeling;
- gebruik operationeel vervolgens een sterke selectie van een of twee drones.

## 6. Is de winst van meerdere drones vooral operationeel of fundamenteel?

Het antwoord is: beide.

### Operationeel

Ja, twee drones helpen direct tijdens detectie, omdat je:

- meer kans hebt op een goede view;
- misses opvangt;
- soms betere fused confidence krijgt.

### Fundamenteel

Ja, viewpoint-diversiteit helpt ook fundamenteel op modelniveau, omdat training op bredere viewpoint-verdelingen beter generaliseert dan training op een smalle viewpoint-beperking.

Dat betekent dat "meer drones" twee verschillende betekenissen heeft:

- meer viewpoints op inferentietijd;
- meer viewpoint-diversiteit op trainingstijd.

Beide zijn waardevol, maar ze moeten methodologisch uit elkaar gehouden worden.

## 7. Wat is het beste experimentele model om dit netjes te bewijzen?

Hier is een experimenteel model dat goed aansluit op de huidige projectstructuur en dat de vraag scherp beantwoordt.

## Laag A: trainingsvraag

Vergelijk drie regimes op exact dezelfde testset:

1. single-view training
2. pair-view training
3. matched-count full-M4 control training

Rapporteer per regime:

- mAP50-95
- mAP50
- F1
- per-class AP50-95

Doel van deze laag:

- isoleren hoeveel winst komt door viewpoint-restrictie;
- en hoeveel door meer data of meer diversity.

## Laag B: operationele vraag

Houd de detector vast en vergelijk:

1. `1-of-1` met 1 drone
2. `1-of-2` met 2 drones
3. eventueel `1-of-3` als diminishing-returns referentie
4. confirmationvarianten zoals `2-of-2` of `2-of-3` als missiebetrouwbaarheid belangrijk is

Rapporteer:

- target detected rate
- target AP50-95
- target strict quality
- target confidence
- per-class gains

Doel van deze laag:

- aantonen wat twee drones operationeel opleveren boven een drone.

## Laag C: complementariteit en relatiepatronen

Vergelijk paren niet alleen op exacte viewpoint-ID, maar ook op:

- azimuth-relatie
- elevation-relatie
- radius-relatie
- mixed diversity types

Gebruik:

- bootstrap over scenes
- matched-scene pairwise differences

Doel van deze laag:

- generaliseerbare ontwerpregels formuleren;
- vermijden dat de thesis alleen leunt op een paar toevallig sterke exacte combinaties met lage support.

## Laag D: fusie

Vergelijk binnen dezelfde selected-view sets:

1. best-view / current method
2. support-weighted OR
3. noisy-OR + best IoU

Doel van deze laag:

- bepalen of de winst van meerdere drones puur een rescue-effect is;
- of dat er ook echte corroboratie tussen detecties is.

## Beste hoofdmetrics voor de thesis

Als centrale metrics zou ik adviseren:

- `target AP50-95`
- `target strict quality`
- `target detected rate`
- `rescue rate given primary miss`
- `complementarity gain vs best single`

Niet als hoofdmetric gebruiken:

- gewone mean AP over losse geselecteerde beelden

Die metric is nuttig als context, maar beantwoordt minder goed de operationele target-vraag.

## 8. Wat is de praktisch beste strategie?

Op basis van de huidige resultaten is de meest verdedigbare praktische strategie:

### Als je maar 1 drone hebt

- kies een sterk, dichtbij gelegen viewpoint;
- vermijd een "gemiddelde" of willekeurige route;
- optimaliseer voor target-kwaliteit, niet voor algemene dekking alleen.

### Als je 2 drones hebt

- laat drone 2 niet hetzelfde doen als drone 1;
- bouw expliciet complementariteit in;
- combineer bij voorkeur verschil in elevation en/of verschil in afstand;
- gebruik drone 2 als rescue-view en indien mogelijk ook voor confidence-fusie.

### Als je een model wilt bouwen voor echte inzet

De meest logische huidige route is:

- train een detector op een brede, diverse viewpoint-set;
- gebruik operationeel twee drones;
- selecteer de beste target-evidence over beide views;
- voeg support-weighted fusie toe als conservatieve deployable upgrade.

Een zwaarder echt multiview-model, zoals geometrische of transformer-gebaseerde multiview fusie, is pas de volgende stap als:

- camera-calibratie beschikbaar is;
- synchronisatie tussen views betrouwbaar is;
- object-ID's of scene-correspondences beter beschikbaar zijn.

## 9. Eindconclusie

De meerwaarde van twee drones ten opzichte van een drone is in dit project duidelijk aantoonbaar.

Die meerwaarde is niet alleen:

- meer kans op een hit,

maar ook:

- betere target-kwaliteit;
- hogere robuustheid bij moeilijke objectklassen;
- echte viewpoint-complementariteit;
- en in beperkte maar reele mate ook fuseerbare cross-view evidentie.

De sterkste wetenschappelijke conclusie is daarom:

`de stap van 1 naar 2 viewpoints is de belangrijkste stap in multiview objectdetectie binnen deze dataset`

en:

`de beste two-drone strategie is niet duplicatie, maar complementaire positionering`

Een derde viewpoint kan nog winst geven, maar de marginale meerwaarde is veel kleiner. Twee drones lijken dus in deze projectresultaten de beste balans te geven tussen extra informatie en afnemende meeropbrengst.
