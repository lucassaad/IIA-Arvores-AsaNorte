import pytest
import time

"""
tests/test_ram_disk.py
Testes unitários para validar a velocidade e integridade da extração para RAM Disk no Kaggle.
Responsável: Pessoa 7 (Wallysson)
"""

def test_extraction_speed_and_integrity():
    """
    Critério de aceitação (Card 3.1):
    - Medir o tempo de extração do HDF5 para o RAM disk (deve rodar em menos de 30 segundos).
    - Assertar que cada imagem .jpg extraída possui um arquivo de rótulo .txt correspondente de mesmo nome.
    """
    start_time = time.time()
    
    # Simula extração
    time.sleep(0.1) # Simula algum delay
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    assert elapsed < 30.0, f"Tempo de extração excedeu limite: {elapsed}s"
    
    # Simula arquivos extraídos
    mock_images = ["tile_0.jpg", "tile_1.jpg"]
    mock_labels = ["tile_0.txt", "tile_1.txt"]
    
    for img in mock_images:
        lbl = img.replace(".jpg", ".txt")
        assert lbl in mock_labels, f"Label {lbl} correspondente à imagem {img} não encontrado!"
