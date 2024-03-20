## 1. domača naloga: glasovanje za pesem Evrovizije
Zahteve zaključka
Odprto: četrtek, 7. marec 2024, 00.00
Rok za oddajo: torek, 26. marec 2024, 00.00
Tisti, ki spremljajo glasovanje za Pesem Evrovizije, pravijo, da ni najbolj objektivno. Predstavniki posameznih držav glasujejo pristransko in favorizirajo nastopajoče iz sorodnih držav. Tako očitno je, da so take vzorce priznali celo organizatorji: v letu 2022 so nekatere države goljufale, zato so njihove glasove zamenjali s podobnimi državami.

Pa preverimo, če res drži! Analizirali bomo pretekla glasovanja. V priloženi datoteki (.xlsx) imamo na voljo podatke glasovanj preteklih tekmovanj. Primerjaj države med sabo tako, da oceniš razdaljo med njimi glede na to, kako glasujejo: glede na njihov profil glasovanja. Pri tem uporabi že razvit postopek za hierarhično razvrščanje v skupine. Za reševanje naloge uporabi zgolj podatke s finalov.

V nalogi boš moral(a) ustrezno rešiti kar nekaj problemov. Na primer, kako zapisati podatke v primerni obliki? Kako združiti podatke iz posameznih let in se pri tem izogniti nepotrebnim povprečjim? Je smiselno obravnavati vse podatke ali jih prefiltriramo? Je res smiselno analizirati vse podatke ali se lahko, ob spreminjajoči se geopolitični situaciji, omejimo na podatke zadnjih nekaj let?

Za oddajo pripravite predstavitvi namenjeno projekcijo, ki jo izvozite kot pdf. Obsega naj le 5 (+1) prosojnic:

- [20%] Podatki. Katere podatke ste analizirali? Kako ste iz podatkov izluščili profile glasovanja? Kako ste ustvarili profil glasovanja posameznih držav? Ste podatke kako obdelali (recimo glede neznanih vrednosti)?
- [20%] Parametri razvrščanja. Kako ste računali razdalje med posameznimi profili ter med posameznimi skupinami? Zakaj ste se odločili za izbrane parametre?
- [20%] Grafični dendrogram (lahko ga implementirate sami ali uporabite iz kake knjižnice; v vsakem primeru morate uporabite rezultate vašega razvrščanja) in smiselno prikazan graf silhuete glede na število skupin. Na dendrogramu označite skupine.
- [20%] Argumentiraj odločitev za izbrane (in prej prikazane) skupine.
- [20%] Razlaga zanimivih skupin. Geopolitični vidiki in analiza glasovanja skupin. Poleg analize na kratko opišite postopek, ki privede do rezultatov glede preferiranih in nepreferiranih držav.

Na poljubno mesto lahko dodate še eno prosojnico s poljubno vsebino.

Oddaja: na spletno učilnico oddajte vašo predstavitev (.pdf) in kodo (eno .py datoteko). Del rešitve je tudi vaša programska koda, zato naj bo pregledna. Vaša koda naj deluje s podano nespremenjeno (.xlsx) datoteko in naj direktno generira vse uporabljene rezultate. Vso obdelavo morate torej narediti v Pythonu.

### Zapiski iz predavanja:
- Cilj: iz danih podatkov pripraviti ustrezen profil države
- Hierarhično gručenje -> zbrat bo treba razdaljo (npr. lahko nad profilom nardiš pca in pol razdaljo med njimi; lahko direktno uporabiš cosinusno razdaljo)
- Grafi;ni prikaz na katerem so izbrane skupine: hočemo ugotoviti ali je vse skupaj samo neki čredno glasovanje (Slovenci vedno glasujejo za to in to državo) ali je vse nakljunčo (ni geopoliticne povezave z glasovanjem)
- 4.prosojnica: mogoče rezervna - pokažeš silhueto in pa če znamo povedat kaj o skupinah oziroma za koga preferenčno skupine glasujejo
- Razbitje po letih ni nujno najbolše, mogoče je to samo za razvrščanje. Mogoce uporabimo povprečno število pik, ki jih je država dala drugi državi ali pa razbij leta na obdobja in potem povprečiš točke
- Število skupin lahko najdeš z najvišjo povprečno silhueto
- Kodo piši sam - se čekira za izvirnost/podobnost, ampak priporoča, da debatiramo
- Zagovor bo powrpoint/pdf pred Markotom (ali in  Zupanom)
- Marko bo tud poganjov našo kodo, zato naj bo lepo dokumentirana - vse more bit tud znotrej enga fajla (python, ne notebook)
- Dendogra lahko izrišemo sami ali pa uporabimo zunanjo knjižnico. Lahko ga pa tudi samo v terminalu z plusi in pomišljaji, če hočemo
- Prašov nas je, če uporablamo copilota in zakaj ne