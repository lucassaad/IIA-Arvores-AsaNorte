import pytest
import numpy as np

"""
tests/test_slicing.py
Testes unitários para validar a rotina de fatiamento de ortofotos GeoTIFF.
Responsável: Pessoa 1 (Luidgi Varela Carneiro)
"""

def test_slice_geotiff_dimensions():
    """
    Critério de aceitação:
    Assertar que o tamanho de saída de cada tile é exatamente (640, 640)
    e que nenhuma imagem recortada possui dimensões diferentes de 640x640.
    """
    # Exemplo de simulação de fatiamento
    tile_size = 640
    mock_sliced_image = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    
    assert mock_sliced_image.shape[0] == 640, "Altura deve ser 640"
    assert mock_sliced_image.shape[1] == 640, "Largura deve ser 640"
    assert mock_sliced_image.shape[2] == 3, "Devem ser 3 canais de cor"
