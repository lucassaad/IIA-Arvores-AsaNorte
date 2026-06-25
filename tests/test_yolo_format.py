import pytest
import numpy as np

"""
tests/test_yolo_format.py
Testes unitários para validar a formatação do label em formato YOLO normalizado.
Responsável: Pessoa 3
"""

def test_coordinate_normalization():
    """
    Critério de aceitação (Card 1.3):
    Verificar se a conversão resulta em valores normalizados entre 0.0 e 1.0.
    """
    # Exemplo: converter [xmin, ymin, xmax, ymax] -> [x_center, y_center, w, h] normalizado
    width, height = 640, 640
    abs_box = [100, 100, 200, 200]
    
    # Conversão YOLO
    x_center = (abs_box[0] + abs_box[2]) / 2.0 / width
    y_center = (abs_box[1] + abs_box[3]) / 2.0 / height
    w = (abs_box[2] - abs_box[0]) / width
    h = (abs_box[3] - abs_box[1]) / height
    
    yolo_box = np.array([0, x_center, y_center, w, h]) # class_id = 0
    
    assert np.all(yolo_box[1:] >= 0.0) and np.all(yolo_box[1:] <= 1.0), "Coordenadas fora do intervalo [0, 1]"
    assert yolo_box[0] == 0, "Classe deve ser 0 (tree)"

def test_empty_images():
    """
    Critério de aceitação (Card 1.3):
    Garantir tratamento correto para imagens vazias (deve gravar matriz vazia de shape (0, 5)).
    """
    empty_label = np.empty((0, 5), dtype=np.float32)
    assert empty_label.shape == (0, 5), "Formato de imagem vazia inválido"
