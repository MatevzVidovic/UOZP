[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/QNg-udn6)
# HW2: Metoda glavnih komponent (PCA)

## Oddaja

Kot oddajo morate svojo kodo dodati na GitHub repozitorij.
Ko vam koda prestane vse teste, ste nalogo opravili.

## Implementacija PCA

V datoteki `hw_pca.py` imate zbrane metode, ki jih morate implementirati. Prosim, preberite opis in se držite tipov, ki jih metoda prejme in vrača. Za testiranje metod smo dodali še datoteko `test_pca.py`.

Metodo glavnih komponent implementirajte kot razred `PCA`, ki deluje po potenčni metodi. 
Metode razreda implementirajte in ne spreminjajte njihovih vhodov.
Predlagamo uporabo funkcij knjižnice `numpy`, zunanjih metod pa ne boste potrebovali.


## 3. domača naloga: PCA in kategorije članov na rtvslo.si
Odprto: torek, 9. april 2024, 00.00
Rok za oddajo: torek, 16. april 2024, 23.59

Ko me zanima, kaj se dogaja, me pogosto zanese na rtvslo.si. Tam objavljajo novice iz zelo različnih področij, ki so lepo razvrščene v kategorije. Ali znamo kategorije kako rekonstruirati?

Prof. Zupan nam je pripravil podatke o več kot 10000 člankih s portala. V priloženi .yaml datoteki je vsak članek predstavljen z naslovom in ključnimi besedami (slednje je določil ChatGPT). V nalogi boste s PCA projicirali podatke v 3D prostor in jih tam poskušali razložiti. Izvedite naslednje:

Ključne besede transformirajte s TF-IDF.
Transformirane podatke sestavite v matriko, ki je primerna za PCA. Pri tem uporabite le ključne besede, ki se pojavijo v vsaj 20 dokumentih.
Izvedite PCA z vašo implementacijo druge naloge. Seveda jo lahko še popravite, ampak tudi popravljena verzija mora zadoščati zahtevam ter testom, ki smo jih postavili tam.
Prikažite projekcijo člankov v interaktivni 3D vizualizaciji, kjer lahko zorni kot premikamo (uporabite vispy). Vizualizacija naj vsebuje tudi graf nateznih koeficientov (loadings plot; sami se odločite, kako ustrezno izbrati značilke zanj).
Nalogo oddate kot pythonski program, ki odpre datoteko rtvslo.yaml v istem direktoriju, izvede zgoraj opisane korake in odpre vizualizacijo. To lahko skupaj traja največ 1 minuto (na mojem 4 leta starem prenosniku, in da, tu je veliko rezerve). Program mora vsebovati tudi PCA in prestati teste 2. naloge tudi, če .yaml ni na voljo.

Oddajte tudi predstavitvi namenjeno projekcijo, ki jo izvozite kot PDF. Obsega naj točno 4 prosojnice (in naj bo brez naslovnice):

1. **Podatki.** Prikažite podatke, na katerih ste izvedli PCA. Prikažite tudi nekaj ilustrativnih vrednosti tabele, ki jim dodate koristno razlago.
2. **Razložena varianca.** Prikažite razloženo varianco po komponentah.
3. **Rezultati.** Prikažite projekcijo in graf nateznih koeficientov.
4. **Interpretacija.** Utemeljite vašo izbiro značilk za graf nateznih koeficientov. Interpretirajte rezultate.

**Oddaja**. Na spletno učilnico oddajte vašo predstavitev (.pdf) in kodo (eno .py datoteko). Del rešitve je tudi vaša programska koda, zato naj bo pregledna.
