# Anleitung zur Nutzung von ControlNet für Image-to-Image Translation
## Datensatz vorbereiten und auf Workstation hochladen
Um Datensätze von einem lokalen Rechner zur Workstation hochzuladen, werden folgende Schritte durchgeführt:
1. **Vorbereitung des Datensatzes**: Stell sicher, dass der Datensatz in einem geeigneten Format vorliegt und alle erforderlichen Dateien enthalten sind. 
   
    Wenn ein neuer Datensatz erstellt werden soll, also beispielsweise neue Bilder hinzugefügt werden sollen, kann das Skript `dataset_tool.py` verwendet werden, das sich im Verzeichnis `scripts` befindet. Beispiel:
   ```bash
    python /image-to-image/ControlNet/scripts/prepare_dataset_for_training.py ` --src path/to/source ` --dst image-to-image/ControlNet/data/real --size 512 --split 0.8 0.1 0.1
    ```

    Das Skript erstellt einen neuen Datensatz im gewünschten Format aus den angegebenen Bildern und speichert ihn im angegebenen Verzeichnis.
    Funktionen:
      - Inventar (Anzahl, Größe)
      - Prüfen und Entfernen korrupten Bilder
      - Konvertieren nach RGB
      - Resize + Pad auf Quadrat-Target
      - Split in train/val/test
      - Manifest (CSV) + Checksummen
  
    Daten aus realen Aufnahmen sollen dabei in das Verzeichnis 'data/real' gespeichert werden, während Simulationsdaten in das Verzeichnis 'data/sim' abgelegt werden sollen.

2. Über WSL/Linux den Datensatz in ein Verzeichnis kopieren, das von der Workstation aus zugänglich ist (z.B. `/mnt/c/Users/YourUsername/Datasets`).
3. **Verbindung zur Workstation herstellen**: SSH verwenden, um eine Verbindung zur Workstation herzustellen.
   ```bash
    ssh username@workstation_ip
    ```
4. Verwende SSH/rsync, um die Dateien vom lokalen Verzeichnis auf die Workstation zu übertragen.
   ```bash
    rsync -avz /mnt/c/Users/YourUsername/Datasets/ username@workstation_ip:/path/to/destination/
    ```
5. **Überprüfung des Uploads**: Nach dem Upload überprüfen, ob alle Dateien korrekt übertragen wurden.

## Modell trainieren 
Um das Modell zu trainieren, können folgende Schritte befolgt werden:
1. **Vorbereitung der Workstation**: Sicherstellen, dass alle erforderlichen Abhängigkeiten und Bibliotheken installiert sind.
2. **Starten des Trainingsskripts**: Das Trainingsskript `train.py` im Verzeichnis `ControlNet` verwenden, um das Modell zu trainieren. Beispiel:
   ```bash
    python ControlNet/train_controlnet.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --controlnet_model_name_or_path lllyasviel/sd-controlnet-canny --train_data_dir ControlNet --caption_column prompt --conditioning_image_column control_image --output_dir ControlNet/model --train_batch_size 4 --num_train_epochs 3 --learning_rate 5e-6 --mixed_precision fp16 --gradient_accumulation_steps 1
    ```
3. **Überwachung des Trainings**: Während des Trainings den Fortschritt überwachen und sicherstellen, dass keine Fehler auftreten.
4. **Speicherung des Modells**: Nach Abschluss des Trainings das Modell im angegebenen Ausgabeordner speichern.

## Modell verwenden für Image-to-Image Translation
Um nun das Modell zu verwenden, um die Image-to-Image Translation durchzuführen, können folgende Schritte befolgt werden:
1. **Vorbereitung der Eingabebilder**: Sicherstellen, dass die Eingabebilder im richtigen Format und im Verzeichnis `inputs` gespeichert sind. 
2. **Starten des Inferenzskripts**: Das Inferenzskript `infer.py` im Verzeichnis `ControlNet/scripts` verwenden, um die Image-to-Image Translation durchzuführen. Beispiel:
   ```bash
    python /image-to-image/ControlNet/scripts/infer.py --model-path /image-to-image/ControlNet/model --input-dir /image-to-image/ControlNet/inputs --output-dir /image-to-image/ControlNet/outputs
    ```
3. **Überprüfung der Ausgabebilder**: Nach Abschluss der Inferenz die generierten Ausgabebilder im Ausgabeordner `outputs` überprüfen.