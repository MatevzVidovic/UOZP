# HW3: Projekcija besedil in metoda glavnih komponent

V tretji domači nalogi boste implementirali nekaj osnovnih metod za projekcijo besedil in metodo glavnih komponent. Implementirane metode boste uporabili na dveh naborih besedilnih podatkov iz prostodostopne spletne knjižnice [Project Gutenberg](https://www.gutenberg.org/).

## Oddaja

Kot oddajo morate svojo kodo in vizualizacije dodati na GitHub repozitorij.
Prosim, da na repozitorij ne nalagate surovih podatkov besedil.
Ko vam koda prestane vse teste, ste nalogo opravili. Pri drugem delu želimo, da se s podatki spoznate in naučite uporabe.

## Del 1: Implementacija metod

V datoteki `helper_functions.py` imate zbrane metode, ki jih morate implementirati. Prosim, preberite opis in se držite tipov, ki jih metoda prejme in vrača. Za testiranje metod smo dodali še datoteko `test_hw3.py`.

### Predprocesiranje besedila

Osnovne metode za predprocesiranje, ki bodo omogočale branje in obdelavo besedil:
- branje dokumentov (`read_text_file`)
- odstranjevanje nepotrebnih znakov (`preprocess_text`)

### Elementi predstavitve besedilnih vrst

Predstavitev besedila v numeričnem kontekstu:
- k-terke znakov (`words_into_kmers`)
- vreča besed (`words_into_bag_of_words`)
- fraze (`words_into_phrases`)
- frekvenca ključev (`term_frequency`)
- inverzna frekvenca v dokumentih (`inverse_document_frequency`)
- transformacija tf-idf (`tf_idf`)


### Metoda glavnih komponent

Metodo glavnih komponent kot razred `PCA`, ki deluje po potenčni metodi.
Metode razreda implementirajte in ne spreminjajte njihovih vhodov.
Dodali smo vam nekaj parametrov v konstruktor razreda, da boste lažje nadzorovali rekurzivni klic potenčne metode.
Predlagamo uporabo funkcij knjižnice `numpy`, zunanjih metod pa ne boste potrebovali.

## Del 2: Projekcija besedil iz zbirke Gutenberg

Drugi del naloge je bolj proste narave. Izbira metod in parametrov je prepuščena vam, tudi če bi želeli katerega od jezikov odstraniti zavoljo boljše vizualizacije.

Pripravili smo vam dve datoteki: `Gutenberg-jeziki.zip` in `Gutenberg-100-zip`. Vaša naloga je z uporabo svojih metod pripraviti dve čimbolj zanimivi vizualizaciji in jih shraniti v datoteki `jeziki.png` in `top100.png`. Za vizualizacijo uporabite knjižnico `matplotlib`.

Za vsako od zbirk narišite sliko razcepa na glavne komponente, kjer na x in y osi ležita prva in druga glavna komponenta. Na obeh oseh označite glavne značilke, ki so imajo v glavni komponenti največje uteži (najpomembnejših 5).
V naslovu slike napišite, katere od metod ste uporabili in s kakšnimi parametri.

### Zbirke besedil 

`Gutenberg-jeziki.zip` je primer zbirke besedil v različnih jezikih, zbranih v svojih mapah. Njihov jezik boste torej vedeli vnaprej.
Pomislite, kakšen način transformacije besedila bo najbolj odražal razlike v jezikih.
Vizualizirajte jih z barvami, lahko pa tudi izpustite katerega od jezikov.

`Gutenberg-100.zip` je primer zbirke besedil v angleškem jeziku, kjer pa o besedilih nimate drugih podatkov. Z uporabo metod poskusite projecirati besedila v nižje-dimenzionalni prostor in pojasnite, na kakšne skupine se razdelijo. Verjetno boste morali za opis skupin dokumente tudi pogledati. Za razliko od prejšnjega primera ne pričakujemo lepih skupin besedil.

Pri interpretaciji in vizualizacijah vam puščamo proste roke, pogoj je le, da svojo vizualizacijo shranite in ustrezno poimenujete. Ko boste vizualizaciji pravilno shranili, se bosta izrisali spodaj.

**Vizualizacija jeziki.png**
<div>
    <img src="jeziki.png" width="500">
</div>

**Vizualizacija top100.png**
<div>
    <img src="top100.png" width="500">
</div>
Ugotovitve:</br>
pg20228.txt in 47629.0.txt nista v angleščini.</br>
Razbral sem tri bolj goste gruče (desna polovica na sredini, naštete od leve proti desni):</br>
1. gruča: imajo več geografskih pojmov (popotovanja, ekspedicije, ...)</br>
2. gruča: bolj mešano - kriminalke, detektivke in razne avanture</br>
3. gruča: romantične knjige</br>
Zelo skupaj so tudi knjige iz iste serije.

