# CycleGAN im Projekt trainieren

Diese Anleitung beschreibt den empfohlenen Ablauf fuer das Training im
Verzeichnis `CycleGAN/pytorch-CycleGAN-and-pix2pix`. Sie bezieht sich auf
unpaare Bilddaten, wie sie fuer `sim2real` verwendet werden.

## 1. Was CycleGAN voraussetzt

CycleGAN lernt zwei Abbildungen ohne Bildpaare:

- `G_A`: Domain A nach Domain B
- `G_B`: Domain B nach Domain A

Die Trainingsdaten muessen deshalb nicht paarweise zusammengehoeren. Beide
Domains sollten aber einen vergleichbaren visuellen Inhalt haben. Eine
beliebige Kombination sehr unterschiedlicher Domains fuehrt meist zu
schlechten Ergebnissen.

## 2. Datensatz vorbereiten

Der Datensatz muss diese Struktur haben:

```text
datasets/sim2real/
|-- trainA/       # Trainingsbilder der Eingabedomain
|-- trainB/       # Trainingsbilder der Zieldomain
|-- testA/        # optional: ungesehene Bilder aus A
`-- testB/        # optional: ungesehene Bilder aus B
```

`trainA` und `trainB` duerfen unterschiedlich viele Bilder enthalten. Der
Loader verwendet pro Epoche die groessere Domain und waehlt Bilder der
anderen Domain standardmaessig zufaellig aus. Die Dateinamen muessen nicht
uebereinstimmen.

Vor dem Training sollten die Daten geprueft werden:

```bash
cd /raid/homes/inf23017/image-to-image/CycleGAN/pytorch-CycleGAN-and-pix2pix
find datasets/sim2real/trainA -type f | wc -l
find datasets/sim2real/trainB -type f | wc -l
find datasets/sim2real/testA -type f | wc -l
find datasets/sim2real/testB -type f | wc -l
```

Das Projekt liest RGB- und Graustufenbilder ein. Fuer normales RGB-Training
bleiben `--input_nc 3` und `--output_nc 3` unveraendert.

## 3. Umgebung einrichten

Das Projekt benoetigt Python 3 sowie eine kompatible PyTorch-/CUDA-
Installation. Die aktuelle Codebasis erwartet PyTorch 2.4 oder neuer.

Fuer eine lokale Conda-Umgebung:

```bash
conda env create -f environment.yml
conda activate <umgebungsname>
pip install dominate visdom wandb
```

Auf dem Cluster werden die vorhandenen Slurm-Skripte mit einem PyTorch-
Container ausgefuehrt. Das Skript `cyclegan_default_train.slurm` mountet das
Projekt nach `/workspace/CycleGAN` und installiert im Container die benoetigten
Pakete. Vorher sollte geprueft werden, ob `nvidia-smi` auf dem zugewiesenen
Knoten funktioniert.

## 4. Empfohlenes Training starten

Ein direkter Lauf aus dem Projektverzeichnis sieht so aus:

```bash
python3 -u train.py \
  --dataroot ./datasets/sim2real \
  --name sim2real_run1 \
  --model cycle_gan \
  --dataset_mode unaligned \
  --netG resnet_9blocks \
  --netD basic \
  --norm instance \
  --gan_mode lsgan \
  --batch_size 1 \
  --lr 0.0002 \
  --lambda_A 10 \
  --lambda_B 10 \
  --lambda_identity 0.5 \
  --n_epochs 100 \
  --n_epochs_decay 100 \
  --save_epoch_freq 5 \
  --no_dropout
```

Die wichtigen Einstellungen sind:

- `--model cycle_gan` aktiviert zwei Generatoren und zwei Diskriminatoren.
- `--dataset_mode unaligned` ist fuer `trainA`/`trainB` erforderlich.
- `--batch_size 1` entspricht dem im Original ueblichen Setup. Groessere
  Werte koennen schneller sein, veraendern aber das Trainingsverhalten und
  benoetigen deutlich mehr GPU-Speicher.
- `--n_epochs` trainiert mit konstanter Lernrate; danach wird sie ueber
  `--n_epochs_decay` linear auf null reduziert.
- `--lambda_A` und `--lambda_B` gewichten die Cycle-Consistency-Verluste.
- `--lambda_identity` hilft, unnoetige Farb- oder Strukturveraenderungen zu
  vermeiden. Bei unterschiedlichen Kanalzahlen muss dieser Wert `0` sein.

Fuer den vorhandenen Sim2Real-Slurm-Lauf kann stattdessen verwendet werden:

```bash
sbatch cyclegan_default_train.slurm
```

Dieses Skript nutzt aktuell `batch_size=8`, `lr=1e-5`,
`lambda_A=lambda_B=11.5`, `lambda_identity=2`, 600 konstante Epochen und
50 Decay-Epochen. Diese Werte sind ein projektspezifischer Startpunkt und
sollten nicht automatisch als beste Konfiguration fuer einen neuen Datensatz
uebernommen werden. Der Experimentname bestimmt das Verzeichnis unter
`checkpoints/`.

## 5. Speicher, Aufloesung und Augmentation

CycleGAN benoetigt vier Netze und ist deshalb speicherintensiv. Standardmaessig
werden Bilder auf 286 Pixel skaliert und zufaellig auf 256 x 256 gecroppt.
`--crop_size` muss bei ResNet-Generatoren durch 4 teilbar sein.

Bei hoher Aufloesung sollte waehrend des Trainings gecroppt werden, zum
Beispiel:

```bash
--preprocess scale_width_and_crop --load_size 1024 --crop_size 360
```

Beim Testen kann anschliessend mit `--preprocess scale_width` auf der groesseren
Aufloesung gearbeitet werden. Fuer rechteckige Bilder kann `--preprocess crop`
im Training und `--preprocess none` im Test sinnvoll sein. Die tatsaechliche
Bildgroesse muss zur Generatorarchitektur passen; `resnet_6blocks` und
`resnet_9blocks` benoetigen Seitenlaengen, die durch 4 teilbar sind.

## 6. Checkpoints und Training fortsetzen

Checkpoints, Optionen und HTML-Visualisierungen liegen unter:

```text
checkpoints/<experimentname>/
```

Ein Training wird mit exakt derselben Konfiguration fortgesetzt:

```bash
python3 -u train.py \
  --dataroot ./datasets/sim2real \
  --name sim2real_run1 \
  --model cycle_gan \
  --continue_train \
  --epoch latest \
  --epoch_count 101 \
  --n_epochs 200 \
  --n_epochs_decay 100 \
  --netG resnet_9blocks \
  --norm instance \
  --no_dropout
```

`--epoch_count` muss zur naechsten zu trainierenden Epoche passen. Bei einem
manuell ausgewaehlten Checkpoint, etwa Epoche 300, muss `--epoch 300` gesetzt
werden. Generatorarchitektur, Normalisierung, Kanalzahl und Dropout duerfen
zwischen Training und Fortsetzung nicht unbemerkt geaendert werden.

## 7. Modelle testen

Beide Richtungen auf dem Testdatensatz testen:

```bash
python3 -u test.py \
  --dataroot ./datasets/sim2real \
  --name sim2real_run1 \
  --model cycle_gan \
  --dataset_mode unaligned \
  --epoch latest \
  --netG resnet_9blocks \
  --norm instance \
  --no_dropout \
  --num_test 50 \
  --results_dir ./results/sim2real_run1
```

Die Ergebnisse und eine HTML-Uebersicht werden unter
`results/sim2real_run1/` gespeichert. Fuer die Bewertung mehrerer Checkpoints
kann `--epoch 300`, `--epoch 400` usw. gesetzt werden.

Nur eine Richtung zu testen spart Speicher. Fuer A nach B:

```bash
python3 -u test.py \
  --dataroot ./datasets/sim2real/testA \
  --name sim2real_run1 \
  --model test \
  --model_suffix _A \
  --epoch latest \
  --netG resnet_9blocks \
  --norm instance \
  --no_dropout \
  --results_dir ./results/sim2real_run1_A
```

Fuer B nach A wird `--dataroot` auf `testB` gesetzt und `--model_suffix _B`
verwendet. Die Richtung kann bei Bedarf mit `--direction BtoA` getauscht
werden.

## 8. Training beurteilen

Die GAN-Losses muessen nicht monoton fallen. Aussagekraeftiger sind regelmaessig
gespeicherte Beispielbilder. Geprueft werden sollten:

1. Werden die typischen Merkmale der Zieldomain sichtbar?
2. Bleiben Geometrie und Inhalt der Eingabe erhalten?
3. Entstehen Farbkipper, starke Artefakte oder Mode Collapse?
4. Sind die Ergebnisse auf `testA`/`testB` besser als auf den Trainingsbildern?

Ein einzelner guter Checkpoint ist nicht zwingend der letzte. Deshalb sollten
mehrere Epochen getestet und anhand einer task-spezifischen Metrik oder einer
visuellen Auswertung verglichen werden. FID allein misst nicht, ob die
Eingabestruktur erhalten bleibt.

## 9. Typische Fehler

- **`empty range for randrange()`**: `--crop_size` ist groesser als ein Bild.
  `--load_size` muss mindestens so gross wie `--crop_size` sein; alternativ
  `resize_and_crop` oder `scale_width_and_crop` verwenden.
- **CUDA-/Speicherfehler**: `--batch_size`, `--crop_size` oder `--load_size`
  reduzieren. Fuer CPU ist `--gpu_ids -1` moeglich, aber das Training ist
  sehr langsam.
- **Fehler beim Laden des Checkpoints**: Beim Testen dieselben Werte fuer
  `--netG`, `--norm`, `--no_dropout`, `--input_nc` und `--output_nc` wie beim
  Training angeben.
- **Keine brauchbaren Ergebnisse trotz sinkender Losses**: Dateninhalt,
  Domainrichtung, Augmentation und mehrere Checkpoints pruefen. CycleGAN ist
  nicht fuer beliebige, semantisch unverbundene Domains geeignet.
- **Keine Visualisierung**: `--no_html` nicht setzen, wenn die HTML-Ausgabe
  benoetigt wird. Fuer Weights & Biases zusaetzlich `--use_wandb` verwenden.

## 10. Praktische Reihenfolge

1. `trainA`, `trainB`, `testA` und `testB` kontrollieren.
2. Einen kurzen Lauf mit wenigen Epochen und kleiner Aufloesung starten.
3. Beispielbilder und Checkpoint-Erzeugung pruefen.
4. Einen laengeren Lauf mit festem Experimentnamen starten.
5. Mehrere Checkpoints auf dem Hold-out-Testset vergleichen.
6. Erst danach die beste Konfiguration fuer weitere Trainingslaeufe oder
   Finetuning verwenden.
