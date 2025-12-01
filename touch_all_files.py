#!/usr/bin/env python3
import os
import subprocess

# Liste von Dateierweiterungen, die wir nicht ändern wollen
skip_extensions = {'.zip', '.jpg', '.jpeg', '.png', '.gif', '.ico', '.lockb', '.so', '.weights', '.onnx', '.mp4', '.avi'}

def touch_file(filepath):
    """Fügt ein Leerzeichen am Ende hinzu, wenn noch keines vorhanden ist"""
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
        
        # Prüfe ob Datei mit Newline endet
        if content and content[-1:] != b'\n':
            with open(filepath, 'ab') as f:
                f.write(b'\n')
        elif not content.endswith(b'\n\n'):
            with open(filepath, 'ab') as f:
                f.write(b'\n')
    except Exception as e:
        print(f"Fehler bei {filepath}: {e}")

def main():
    root = '.'
    count = 0
    
    for root_dir, dirs, files in os.walk(root):
        # Überspringe .git Verzeichnis
        if '.git' in root_dir:
            continue
            
        for file in files:
            filepath = os.path.join(root_dir, file)
            
            # Überspringe bestimmte Dateitypen
            _, ext = os.path.splitext(file)
            if ext.lower() in skip_extensions:
                continue
            
            # Überspringe bereits geänderte Dateien
            if filepath == './.gitignore':
                continue
                
            try:
                touch_file(filepath)
                count += 1
                if count % 50 == 0:
                    print(f"{count} Dateien bearbeitet...")
            except Exception as e:
                print(f"Überspringe {filepath}: {e}")
    
    print(f"Insgesamt {count} Dateien bearbeitet")

if __name__ == '__main__':
    main()

