import os
import cv2
import h5py
import numpy as np

"""
utils.py - Funções utilitárias compartilhadas do projeto projeto2-iia.
Dividido entre as necessidades de pipeline de dados, pseudo-labelling e carregamento no Kaggle.
"""

# =====================================================================
# NÚCLEO 1: ENGENHARIA DE DADOS & PIPELINE
# =====================================================================

def slice_geotiff(input_path, output_dir, tile_size=640):
    """
    Fatia uma ortofoto GeoTIFF em blocos (tiles) de tamanho tile_size x tile_size.
    Implementado por: Pessoa 1
    """
    # TODO: Implementar fatiamento usando a biblioteca rasterio e Window.
    pass

def carregar_imagem_jpg(img_path):
    """
    Carrega imagem do disco.
    """
    # Exemplo simples de leitura
    img = cv2.imread(img_path)
    return img

def criar_hdf5_bruto(hdf5_path, image_paths):
    """
    Reordena canais GeoTIFF e grava no formato compilado HDF5 inicial.
    Implementado por: Pessoa 2
    """
    with h5py.File(hdf5_path, 'w') as f:
        # Criamos dois subgrupos principais
        grupo_images = f.create_group('images')
        grupo_labels = f.create_group('labels') # Vazio inicialmente
        
        for idx, img_path in enumerate(image_paths):
            # Carrega a imagem e garante tamanho 640x640x3 do tipo uint8
            img_data = carregar_imagem_jpg(img_path) 
            
            if img_data is None:
                continue
                
            # Salva no HDF5 com compressão gzip balanceada (nível 4)
            grupo_images.create_dataset(
                name=f"tile_{idx}",
                data=img_data,
                compression="gzip",
                compression_opts=4,
                chunks=(640, 640, 3) # Otimiza leitura individual de cada imagem
            )
            
            # Cria dataset de label correspondente vazio (classe -1 ou vazio)
            # Formato: matriz Nx5 para caixas YOLO [class_id, x_center, y_center, w, h]
            grupo_labels.create_dataset(
                name=f"tile_{idx}",
                data=np.empty((0, 5), dtype=np.float32)
            )

def salvar_pseudo_labels(hdf5_path, tile_name, boxes):
    """
    Atualiza as pseudo-labels geradas pelo DeepForest no arquivo HDF5.
    boxes: numpy array shape (N, 5) com as coordenadas normalizadas.
    Implementado por: Pessoa 3
    """
    with h5py.File(hdf5_path, 'r+') as f: # Modo leitura/escrita
        # Deleta a label vazia anterior e grava a nova
        if tile_name in f['labels']:
            del f['labels'][tile_name]
        f['labels'].create_dataset(name=tile_name, data=boxes, dtype=np.float32)


# =====================================================================
# NÚCLEO 2: CURADORIA, CONTROLE DE QUALIDADE & DIVISÃO DE DADOS
# =====================================================================

def exportar_hdf5_para_roboflow(hdf5_path, output_dir):
    """
    Extrai as imagens e labels do HDF5 em pastas locais temporárias
    para upload no Roboflow.
    Implementado por: Pessoa 4
    """
    # TODO: Implementar extração local para upload.
    pass


# =====================================================================
# NÚCLEO 3: MODELAGEM, TREINAMENTO & AVALIAÇÃO
# =====================================================================

def extrair_hdf5_para_ram(hdf5_path, output_dir="/dev/shm/dataset"):
    """
    Lê o HDF5 e extrai diretamente no RAM disk (/dev/shm) para treinamento YOLO acelerado.
    Implementado por: Pessoa 7
    """
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)
    
    with h5py.File(hdf5_path, 'r') as f:
        images = f['images']
        labels = f['labels']
        
        for name in images.keys():
            # 1. Recupera os pixels da imagem e salva como JPG na RAM
            img_array = images[name][:]
            cv2.imwrite(f"{output_dir}/images/{name}.jpg", img_array)
            
            # 2. Recupera os boxes e salva em TXT formato YOLO na RAM
            box_array = labels[name][:]
            txt_path = f"{output_dir}/labels/{name}.txt"
            with open(txt_path, 'w') as txt_file:
                for box in box_array:
                    # Formato YOLO: class_id x_center y_center width height
                    txt_file.write(f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
