# NInfer-M1, in italiano

### Un modello da 180 miliardi di parametri su un MacBook da 64 GB

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) al vero Q4: un file da **97,5 GB**, una volta e mezza
la memoria del Mac, a **circa 30 token/s**, **29 token/s dopo un prompt da 28.000 token**, con **ogni token controllato dal
modello intero**.

![stesso file, stesso Mac](images/engines.png)

Il modello è un misto di esperti: 125 miliardi di parametri di cui solo 6 attivi per ogni token, più una tabella di
embedding n-gram da 51 miliardi. Al Q4 il file pesa 97,5 GB; il Mac ha 64 GB. NInfer-M1, un fork di llama.cpp tarato per
l'M1, tiene gli esperti più usati in una cache Metal da 34 GiB, **legge gli altri dall'SSD mentre la GPU lavora**, riaccende
l'attenzione sparsa che il GGUF aveva perso, propone bozze con la testa MTP del modello. **Ogni token proposto è controllato dal modello
intero**, quindi niente è approssimato: con le bozze MTP il testo greedy è uguale a quello della decodifica normale in tutte
le richieste confrontate.

È un server compatibile con OpenAI e Anthropic, ed è il backend locale quotidiano di Claude Code su questo Mac da ottobre 2026.

## I numeri

Stesso file, stesso Mac, stessi prompt, un motore alla volta (M1 Max, 64 GB, greedy, 14 ottobre 2026):

| | prompt brevi | dopo un prompt da 28k token | lettura di quel prompt |
|---|---:|---:|---:|
| **NInfer-M1** | **30,9 tok/s** | **29,1 tok/s** | **2 min 08 s** (220 tok/s) |
| fork llama.cpp con lo streaming (mihailescu2m) | 16,0 | 13,2 | 6 min 56 s (68 tok/s) |
| llama.cpp master, `--cpu-moe` | 9,7 | 4,8 | 19 min 38 s (24 tok/s) |

**1,9× e 3,2× sui prompt brevi, 2,2× e 6,1× con 28k token di contesto, e il prompt si legge 3,3× e 9,2× più in fretta.**
Anche senza bozze NInfer-M1 fa 22,8 tok/s contro 9,7. (Misurato con una bozza MTP con sotto-testa da 64k token; la bozza
standard dell'avvio rapido dà 29,7 tok/s sui prompt brevi, il 4% in meno.)

- **Il contesto lo rallenta poco:** ~29 tok/s a 1,5k token, ~31 a 15k, **27 a 65k** (con attenzione densa: 11).
- **Stesse risposte:** su dodici compiti valutati il testo è identico a quello del fork in 12 casi su 12 e a quello di
  llama.cpp master in 11 su 12 (il dodicesimo differisce negli spazi di un oggetto JSON), e tutti passano gli stessi nove.
  Sulle risposte più lunghe due motori qualsiasi possono separarsi su un token quasi in parità, come succede con l'aritmetica
  in virgola mobile.
- **Un modello da 180 B letto dall'SSD va quanto il miglior motore MLX su un 27 B che sta in memoria**, dopo un prompt da
  28k token: 29,1 contro 28,6 tok/s, e legge il prompt 1,6× più in fretta. Quando il modello *sta* in memoria MLX è più
  veloce sui prompt brevi (MTPLX: 37 tok/s sul 27 B); NInfer-M1 serve per i modelli che non ci stanno.
- **Sessioni lunghe da agente:** due conversazioni da 30k token che si alternano rispondono in **5,6 s invece di 140 s**
  (quella uscita dallo slot torna dal disco in 0,3 s); le conversazioni sopravvivono al riavvio del server.
- **Sicuro su un portatile:** 51,9 GiB di memoria bloccata a ogni lunghezza di contesto fino a 65k.

Metodo, avvertenze, opzioni di ogni motore e numeri grezzi: [BENCHMARKS](BENCHMARKS.md) (in inglese).

## Come fa

- **Esperti dall'SSD, in modo esatto.** 286 dei 512 esperti di ogni strato stanno in una cache Metal da 34 GiB; un passo
  della CPU dentro il flusso di comandi della GPU legge quelli mancanti con 9 thread e riscrive gli indici. Il router di
  ogni strato gira *prima* dell'attenzione per far partire le letture in anticipo. Stesso risultato che con tutti gli
  esperti in memoria.
- **L'attenzione sparsa, riaccesa.** Il GGUF Q4 salva il rapporto di compressione 0, quindi llama.cpp guardava tutto il
  contesto. NInfer-M1 ripristina l'attenzione del modello (ogni query legge i suoi 2.048 token migliori), a pezzi per
  stare in memoria e deterministica: **decodifica ×2,5 a 65k token**.
- **Metal specifico per l'M1.** Sottomissione Metal 4 per i grafi con passi sulla CPU, prodotti matrice a poche righe con
  soglie per l'M1, kernel per le hyper-connection.
- **Bozze.** La testa MTP propone 3 token con una sotto-testa da 64k token; bozze n-gram fino a 32 token per le modifiche.
- **Pensato per sessioni lunghe.** Un secondo livello della cache dei prompt su disco.

## Avvio rapido

Serve un Mac Apple Silicon con **64 GB** (tarato su M1 Max), **macOS 26** consigliato (la sottomissione Metal 4 vale ~20%),
e **~100 GB liberi sull'SSD interno**: un disco esterno è troppo lento per le letture degli esperti.

```bash
# il motore: il binario pronto dalla pagina Releases, oppure
git clone https://github.com/nervitesi/NInfer-M1-Flash && cd NInfer-M1-Flash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_OPENSSL=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build build --target llama-server -j

# i pesi (non sono in questo repository)
hf download mradermacher/Qwen3.8-Flash-Next-i1-GGUF Qwen3.8-Flash-Next.i1-IQ4_XS.gguf --local-dir ~/models
hf download unsloth/Qwen3.8-Flash-Next-GGUF MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --local-dir ~/models

# si avvia
MODEL=~/models/Qwen3.8-Flash-Next.i1-IQ4_XS.gguf \
DRAFT=~/models/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
scripts/ninfer-m1/run-flash-next-q4.sh
```

Il server ascolta su `127.0.0.1:8080`. Le prime richieste sono più lente (24–26 tok/s): la cache degli esperti parte vuota e
si riempie dall'SSD. Lascia circa 10 GB di memoria al sistema e non alzare la cache degli esperti sopra 34 GiB.

## Limiti onesti

- Costruito e misurato su **una macchina** (MacBook Pro 16" 2021, M1 Max, 64 GB, macOS 26); gli altri chip Apple Silicon
  non sono provati.
- È un **fork** di un progetto che corre veloce, tarato per **un modello**. Sui modelli densi che stanno in memoria è
  circa il 3% dietro llama.cpp master, e i motori MLX vanno ancora più forte.
- La speculazione è esatta nell'algoritmo (un token proposto resta solo se lo sceglie il modello), non al bit in virgola
  mobile: se cambia la forma del blocco di verifica un token quasi in parità può cambiare (la bozza standard cambia una delle
  tre risposte da 256 token dopo 922 caratteri). Controllato su richieste greedy.
- I benchmark lasciano fuori le bozze n-gram (ripetendo un prompt ripetono la risposta precedente); l'avviatore le tiene,
  perché velocizzano le modifiche (`NGRAM=0` le spegne e dà la configurazione delle tabelle).

Il diario completo del lavoro (in italiano, ~3.400 righe, con ogni fallimento) resta nel repository privato di laboratorio;
la sintesi, in inglese, è in [JOURNEY](JOURNEY.md).

Licenza MIT, come llama.cpp. Non affiliato al team Qwen, a ggml-org, ad Apple né al motore CUDA
[ninfer](https://github.com/Neroued/ninfer). Crediti e avvisi di terze parti: [NOTICE](NOTICE.md).
