# Скрипт для импорта товаров в WooCommerce

Данный скрипт предназначен для автоматизированного импорта товаров со стороннего сайта в интернет-магазин WooCommerce. 

---

## Возможности

- **Парсинг данных о товарах**:
  - Сбор информации с сайта, включая название товара, описание, цену, изображения, и дополнительные характеристики.
  - Распознавание структуры категорий товаров.
  - Автоматическое создание категорий в WooCommerce (включая вложенные категории).

- **Интеграция с WooCommerce**:
  - Импорт товаров с использованием WooCommerce API.
  - Установка статуса наличия товара (`instock` или `outofstock`).
  - Поддержка пакетной загрузки товаров для повышения производительности.

---

## Какие данные собираются по товару

Каждый товар включает в себя следующие параметры:

1. **Название товара** – Полное название продукта.  
2. **Описание** – Подробная информация о товаре.  
3. **SKU** – Уникальный идентификатор товара.  
4. **Цена** – Регулярная цена товара с возможностью её модификации (например, добавление наценки).  
5. **Изображения** – Ссылки на изображения товаров, которые автоматически загружаются в WooCommerce.  
6. **Категория** – Привязка товара к соответствующей категории. Если категория отсутствует, она создается.  
7. **Статус наличия** – Определяется автоматически в зависимости от информации на сайте (в наличии или под заказ).  
8. **Атрибуты товара** – Характеристики, такие как размеры, цвета и другие параметры.


---
Собрать Docker-образ для модели LLaMA с поддержкой RAG (Retrieval-Augmented Generation) на Ubuntu с нуля можно по следующей инструкции. Здесь будет использоваться Hugging Face Transformers и FAISS для работы с RAG.

1. Подготовка системы
Обновите систему и установите необходимые пакеты:

bash
Копировать
Редактировать
sudo apt update && sudo apt upgrade -y
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common git
Установите Docker и Docker Compose:

bash
Копировать
Редактировать
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
Убедитесь, что Docker установлен:

bash
Копировать
Редактировать
docker --version
2. Создание проекта
Создайте рабочую директорию:

bash
Копировать
Редактировать
mkdir llama-rag-docker && cd llama-rag-docker
Создайте файл Dockerfile для модели LLaMA.

3. Dockerfile для LLaMA
Создайте файл Dockerfile:

```Dockerfile
# Используем легковесный образ Python
FROM python:3.9-slim

# Установим системные зависимости
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libopenblas-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Установим Python-зависимости
RUN pip install --upgrade pip
RUN pip install torch transformers faiss-cpu flask

# Клонируем репозиторий с LLaMA
WORKDIR /app
RUN git clone https://github.com/facebookresearch/llama.git
WORKDIR /app/llama
RUN pip install -r requirements.txt

# Копируем серверный скрипт в контейнер
COPY server.py /app/server.py

# Открываем порт для API
EXPOSE 8080

# Запускаем сервер
CMD ["python", "/app/server.py"]
```

4. Скрипт для API
Создайте файл server.py для запуска API:

```python
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Инициализация Flask
app = Flask(__name__)

# Загружаем модель LLaMA
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Укажите вашу модель
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
```

5. Сборка Docker-образа
Соберите Docker-образ:

bash
Копировать
Редактировать
docker build -t llama-rag .
6. Запуск контейнера
Запустите контейнер:

bash
Копировать
Редактировать
docker run -d -p 8080:8080 llama-rag
Теперь ваш API доступен на порту 8080.

7. Добавление RAG через FAISS
Для добавления функционала RAG измените файл server.py:

Добавьте библиотеку FAISS:

bash
Копировать
Редактировать
pip install faiss-cpu
Обновите код API:

```python
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import torch

# Flask API
app = Flask(__name__)

# LLaMA модель
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# FAISS индексация
dimension = 768  # Размерность эмбеддингов (проверьте для вашей модели)
index = faiss.IndexFlatL2(dimension)

# Функция добавления документа
def add_document(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
    embeddings = model.get_input_embeddings()(inputs["input_ids"]).mean(dim=1).cpu().detach().numpy()
    index.add(embeddings)

@app.route("/add_document", methods=["POST"])
def add_document_api():
    text = request.json.get("text", "")
    add_document(text)
    return jsonify({"status": "Document added"})

@app.route("/search", methods=["POST"])
def search():
    query = request.json.get("query", "")
    inputs = tokenizer(query, return_tensors="pt", truncation=True).to(model.device)
    query_embedding = model.get_input_embeddings()(inputs["input_ids"]).mean(dim=1).cpu().detach().numpy()
    distances, indices = index.search(query_embedding, k=5)
    return jsonify({"distances": distances.tolist(), "indices": indices.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
```
8. Пересоберите образ
Обновите образ, чтобы включить изменения:

bash
Копировать
Редактировать
docker build -t llama-rag .
9. Тестирование
Добавьте документ:

bash
Копировать
Редактировать
curl -X POST -H "Content-Type: application/json" \
     -d '{"text": "LLaMA — это языковая модель от Meta."}' \
     http://localhost:8080/add_document
Выполните запрос:

bash
Копировать
Редактировать
curl -X POST -H "Content-Type: application/json" \
     -d '{"query": "Что такое LLaMA?"}' \
     http://localhost:8080/search
Вы получите ближайшие результаты из индекса.

  File "/usr/local/lib/python3.9/site-packages/huggingface_hub/utils/_http.py", line 423, in hf_raise_for_status
    raise _format(GatedRepoError, message, response) from e
huggingface_hub.errors.GatedRepoError: 401 Client Error. (Request ID: Root=1-678cb2b1-40b8db10502f81c51cf838d6;d07d83a3-98c7-4da8-8a65-c2db11e661eb)

Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.2-3B/resolve/main/config.json.
Access to model meta-llama/Llama-3.2-3B is restricted. You must have access to it and be authenticated to access it. Please log in.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/app/server.py", line 10, in <module>
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  File "/usr/local/lib/python3.9/site-packages/transformers/models/auto/tokenization_auto.py", line 891, in from_pretrained
    config = AutoConfig.from_pretrained(
  File "/usr/local/lib/python3.9/site-packages/transformers/models/auto/configuration_auto.py", line 1054, in from_pretrained
    config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
  File "/usr/local/lib/python3.9/site-packages/transformers/configuration_utils.py", line 591, in get_config_dict
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
  File "/usr/local/lib/python3.9/site-packages/transformers/configuration_utils.py", line 650, in _get_config_dict
    resolved_config_file = cached_file(
  File "/usr/local/lib/python3.9/site-packages/transformers/utils/hub.py", line 421, in cached_file
    raise EnvironmentError(
OSError: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/meta-llama/Llama-3.2-3B.
401 Client Error. (Request ID: Root=1-678cb2b1-40b8db10502f81c51cf838d6;d07d83a3-98c7-4da8-8a65-c2db11e661eb)

Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.2-3B/resolve/main/config.json.
Access to model meta-llama/Llama-3.2-3B is restricted. You must have access to it and be authenticated to access it. Please log in.


