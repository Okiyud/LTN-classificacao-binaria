# Classificação Binária Cachorro vs Gato com Logic Tensor Networks (LTN)

Sistema de classificação neuro-simbólico que combina redes neurais convolucionais com raciocínio lógico para distinguir entre imagens de cachorros e gatos.

## Integrantes do Projeto

- André Yudji Silva Okimoto
- Carolina Falabelo Maycá
- Fernando Lucas Almeida Nascimento
- Guilherme Dias Correa
- Guilherme Louro de Salignac Souza
- Luiza da Costa Caxeixa
- Nicolas Mady Correa Gomes
- Sofia de Castro Sato

---
##  Configuração do Ambiente

### Pré-requisitos

```bash
# Python 3.8+
python --version

# Anaconda/Miniconda (recomendado)
conda --version
```

### Instalação de Dependências

```bash
# Instalar bibliotecas essenciais
pip install torch torchvision
pip install LTNtorch
pip install matplotlib numpy
pip install jupyter notebook

# Ou usando conda
conda install pytorch torchvision -c pytorch
conda install matplotlib numpy jupyter
pip install LTNtorch
```

---

## Estrutura do Projeto

```
LTN-classificacao-binaria/
├── tutorial_dogs_cats.ipynb    # Notebook principal
├── PetImages/                  # Dataset
│   ├── Cat/                   # Imagens de gatos
│   └── Dog/                   # Imagens de cachorros
├── README.md                  # Este arquivo
└── kagglecatsanddogs_5340/    # Dataset alternativo
```

---

## 🚀 Como Executar

### 1. Preparar o Dataset

```bash
# Verificar estrutura do dataset
ls -la PetImages/
# Deve conter: Cat/ e Dog/

# Verificar quantidade de imagens
echo "Gatos: $(ls PetImages/Cat/ | wc -l)"
echo "Cachorros: $(ls PetImages/Dog/ | wc -l)"
```

### 2. Executar o Notebook

```bash
# Iniciar Jupyter
jupyter notebook

# Abrir: tutorial_dogs_cats.ipynb
# Executar células sequencialmente
```

### 3. Configurações de Treinamento

```python
# Principais parâmetros (editáveis no notebook)
RESOLUCAO_IMAGEM = 64    # Tamanho das imagens
BATCH_SIZE = 32          # Tamanho do lote
EPOCAS_TOTAL = 40        # Número de épocas
TAXA_APRENDIZADO = 0.001 # Learning rate
```

---
