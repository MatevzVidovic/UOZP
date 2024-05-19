# 5. domača naloga - Napovedovanje števila komentarjev na spletnem poratlu rtvslo.si
Rok za oddajo: torek, 14. maj 2024, 23.59
V sklopu domače naloge boste zgradili model, ki bo napovedoval število komentarjev pod članki na spletnem portalu RTVSlo.

## Podatki
Podatke že poznate iz prejšnje naloge. Tokrat so v malo drugačni in razširjeni obliki:
~~~
{
  "url": "Povezava do prispevka",
  "authors": ["Seznam avtorjev"],
  "date": "Čas objave",
  "title": "Naslov prispevka",
  "paragraphs": ["Seznam odstavkov"],
  "figures": [
    {
      "caption": "Podnaslov slike",
      "img": "Povezava do slike",
      "source": "Vir",
      "caption.en": "Podnaslov slike v angleščini"
    },
    ...
    ],
  "lead": "Uvodni tekst",
  "topics": "Kategorija prispevka",
  "keywords": ["Ključne besede prispevka"],
  "gpt_keywords": ["Ključne besede generirane s pomočjo Chat-GPT"],
  "n_comments": "Število komentarjev",
  "id": "Identifikator prispevka",
},
~~~
Vse prispevke smo zapakirali v priloženo (na tem naslovu) datoteko ```rtvslo_train.json.gzip```. Trenutno so vam na voljo članki za obdobje od 20.4.2023 do 3.4.2024. Datoteko bomo najbrž vmes tudi kaj posodobili z novejšimi prispevki. O tem vas bomo pravočasno obvestili na Slacku!

Primer branja datoteke je v skripti ```hw5.py```.

## Navodila
Kako se lotite analize, je veliki meri prepuščeno vam. Edina izjema je izbira napovednega modela, ki mora biti linearna regresija. Pri tem lahko značilke pripravite na poljuben način, tako da so seveda dovoljene tudi različne polinomske regresije.

Če dobro zadovoljite spodaj naštete kriterije, boste nalogo opravili. Pri tem je pomembno, da znate uporavičiti vse korake analize. Nalogo bomo ocenjevali iz različnih vidikov:

1. **Ustrezna predpriprava podatkov.**: Ste uspešno pripravili podatke v obliko primerno za učenje modela? Katere značilke ste izbrali, na novo ustvarili in ali jih znate obrazložiti?

2. **Vrednotenje modela.** Kako ste ovrednotili točnost modela? Kakšno točnost pričakujete na novih podatkih?

3. **Razlaga modela.** Razložite, kako vaš zgrajen model napoveduje. Katere značilke zato (najbolj) uporablja in kako vrednosti le-teh vplivajo na napovedi.

4. **Modela na novih podatkih.** Po oddaji bomo vaš model uporabili na novih podatkih, kot bi ga sicer uporabili v praksi. Ali se rezultati vašega vrednotenja dobro ujemajo s točnostjo na novih podatkih?

In še **bonus točke**. Kako točen je vaš model? Da bo naloga še bolj zanimiva, bomo modelom z najboljšimi rezultati na novih podatkih podelili dodatne točke.

## Oddaja
Prosimo vas, da končni model implementirate v skripto ```hw5.py```. Skripta nam bo omogočila, da bomo vaš model preizkusili na novih podatkih. V skripti dopolnite funkciji ```RTVSlo.fit``` in ```RTVSlo.predict```.

Torej, v ```hw5.py``` implementirajte le končni model, ki ga boste oddali. Vse ostale analize, vrednotenje ali iskanje najboljših parametrov, pa izvajajte ločeno (lahko tudi v notebook obliki). Vso kodo pa vendarle priložite v tale repozitorij. Del rešitve je tudi vaša programska koda, zato naj bo pregledna. Učnih podatkov ne nalagajte v Github repozitorij, da ga ohranite majhnega.

V ```hw5.py``` je dovoljena le uporaba knjižnic, ki smo jih podali v ```requirements.txt```. Algoritmov za obdelavo in modelov vam ni treba implementirati. Enostavno lahko uporabite katero od knjižnic, ki vam te metode že ponuja (npr. scikit-learn). Priporočamo, da uporabljene metode poznate. Bodite pozorni, kako si boste pripravili okolje. Če vse deluje OK, lahko preverite tako, da poženete ```test_hw5.py``` (ali pa pogledate, če se vse požene normalno na Github Classroom).

Na učilnici oddajte tudi predstavitvi namenjeno projekcijo, ki jo izvozite kot PDF. Obsega naj največ 4 prosojnice in naj vam bo v pomoč pri zagovoru vaše rešitve.

Za pristop k nalogi in oddajo na Github Classroom uporabite sledečo povezavo: https://classroom.github.com/a/doKWcl1f