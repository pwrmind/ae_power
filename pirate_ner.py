import sys
import os
import re
import json
from datasets import load_dataset

# Константы геометрии внимания 1/4^x
WING_WEIGHTS = {0: 1.0, 1: 0.25, 2: 0.0625, 3: 0.015625, 4: 0.003906, 5: 0.000976}
MODEL_FILE = "ner_model.json"

# Список стоп-слов, которые физически НЕ МОГУТ быть именами, городами или организациями
STOP_WORDS = {
    "вчера", "сегодня", "завтра", "в", "на", "с", "из", "к", "по", "о", "об", "за", "под", "над", "пред", "для", "при",
    "и", "а", "но", "да", "или", "что", "чтобы", "как", "где", "когда", "почему", "зачем", "это", "этот", "эта", "эти",
    "он", "она", "оно", "они", "мы", "вы", "я", "ты", "мой", "твой", "свой", "его", "ее", "их", "был", "была", "было", "были"
}

def tokenize(text):
    text = text.lower()
    text = text.replace("санкт-петербург", "санктпетербург")
    return re.findall(r'\w+|[^\w\s]', text)

def get_suffix(word):
    return f"suff_{word[-3:]}" if len(word) >= 3 else None

# =====================================================================
# КОМАНДА 1: ОБУЧЕНИЕ И СОХРАНЕНИЕ МОДЕЛИ (15 000 СТРОК)
# =====================================================================
def train_and_save():
    print("[1/3] Загрузка масштабного датасета WikiAnn (Hugging Face)...")
    # Берем 15 000 предложений для создания сверхплотного графа знаний
    raw_dataset = load_dataset("wikiann", "ru", split="train[:15000]") 
    
    tags_features = raw_dataset.features["ner_tags"].feature.names
    id_to_tag = {i: name for i, name in enumerate(tags_features)}
    
    token_counts = {}
    tag_counts = {tag: 0 for tag in tags_features}
    graph_left_context = {}  
    graph_right_context = {} 
    graph_suffix_to_tag = {} 
    
    print("[2/3] Построение разреженных матриц позиционного контекста...")
    for item in raw_dataset:
        tokens = [t.lower() for t in item["tokens"]]
        tags = [id_to_tag[tag_id] for tag_id in item["ner_tags"]]
        
        if len(tokens) != len(tags): 
            continue
            
        for i, token in enumerate(tokens):
            tag = tags[i]
            token_counts[token] = token_counts.get(token, 0) + 1
            tag_counts[tag] += 1
            
            suff = get_suffix(token)
            if suff:
                if suff not in graph_suffix_to_tag: graph_suffix_to_tag[suff] = {}
                graph_suffix_to_tag[suff][tag] = graph_suffix_to_tag[suff].get(tag, 0) + 1
                
            for dist in range(1, 6):
                if i - dist >= 0:
                    prev_word = tokens[i - dist]
                    if prev_word not in graph_right_context: graph_right_context[prev_word] = {}
                    if dist not in graph_right_context[prev_word]: graph_right_context[prev_word][dist] = {}
                    graph_right_context[prev_word][dist][tag] = graph_right_context[prev_word][dist].get(tag, 0) + 1
                
                if i + dist < len(tokens):
                    next_word = tokens[i + dist]
                    if next_word not in graph_left_context: graph_left_context[next_word] = {}
                    if dist not in graph_left_context[next_word]: graph_left_context[next_word][dist] = {}
                    graph_left_context[next_word][dist][tag] = graph_left_context[next_word][dist].get(tag, 0) + 1

    model_data = {
        "token_counts": token_counts,
        "tag_counts": tag_counts,
        "graph_left_context": graph_left_context,
        "graph_right_context": graph_right_context,
        "graph_suffix_to_tag": graph_suffix_to_tag
    }
    
    print(f"[3/3] Сохранение весов в автономный файл '{MODEL_FILE}'...")
    with open(MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump(model_data, f, ensure_ascii=False, indent=2)
    print("--- Обучение на 15 000 строк завершено! Скрипт готов к бою. ---")

# =====================================================================
# КОМАНДА 2: ИНФЕРЕНС (РАСПОЗНАВАНИЕ СУЩНОСТЕЙ)
# =====================================================================
def load_model():
    if not os.path.exists(MODEL_FILE):
        print(f"Ошибка: Модель не найдена. Выполните: uv run pirate_ner.py train")
        sys.exit(1)
    with open(MODEL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def predict_tags(tokens, model):
    predicted_tags = []
    tag_counts = model["tag_counts"]
    
    for target_idx, target_token in enumerate(tokens):
        # ДОРАБОТКА: Если токен в списке стоп-слов или является числом/пунктуацией, жестко даем ему тег 'O'
        if target_token in STOP_WORDS or not target_token.isalnum() or target_token.isdigit():
            predicted_tags.append("O")
            continue
            
        tag_scores = {tag: 0.0001 for tag in tag_counts}
        
        # 1. Морфология (суффиксы)
        suff = get_suffix(target_token)
        if suff and suff in model["graph_suffix_to_tag"]:
            for tag, count in model["graph_suffix_to_tag"][suff].items():
                tag_scores[tag] += (count / tag_counts[tag]) * 1.5
        
        # 2. Поля внимания (5 шагов)
        for context_idx, context_token in enumerate(tokens):
            if context_token == target_token: continue
            distance = abs(target_idx - context_idx)
            wing_weight = WING_WEIGHTS.get(distance, 0.0)
            
            if wing_weight > 0:
                if context_idx < target_idx:
                    if context_token in model["graph_right_context"] and str(distance) in model["graph_right_context"][context_token]:
                        for tag, count in model["graph_right_context"][context_token][str(distance)].items():
                            tag_scores[tag] += wing_weight * (count / tag_counts[tag])
                elif context_idx > target_idx:
                    if context_token in model["graph_left_context"] and str(distance) in model["graph_left_context"][context_token]:
                        for tag, count in model["graph_left_context"][context_token][str(distance)].items():
                            tag_scores[tag] += wing_weight * (count / tag_counts[tag])

        best_tag = max(tag_scores, key=tag_scores.get)
        
        # ДОРАБОТКА: Порог уверенности модели. Если суммарный заряд слишком мал, сбрасываем в 'O'
        if all(v == 0.0001 for v in tag_scores.values()) or tag_scores[best_tag] < 0.005:
            best_tag = "O"
            
        predicted_tags.append(best_tag)
    return predicted_tags

def process_text(text):
    model = load_model()
    tokens = tokenize(text)
    tags = predict_tags(tokens, model)
    
    print(f"\nРезультат анализа:")
    for token, tag in zip(tokens, tags):
        if tag != "O":
            print(f"  [{tag}] -> {token.upper()}")
        else:
            print(f"  {token}")

def process_file(file_path):
    model = load_model()
    if not os.path.exists(file_path):
        print(f"Ошибка: Файл '{file_path}' не найден.")
        return
        
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    output_lines = []
    for line in lines:
        if not line.strip(): 
            output_lines.append("\n")
            continue
        tokens = tokenize(line)
        tags = predict_tags(tokens, model)
        
        labeled_line = []
        for token, tag in zip(tokens, tags):
            if tag != "O":
                labeled_line.append(f"{token}({tag})")
            else:
                labeled_line.append(token)
        output_lines.append(" ".join(labeled_line) + "\n")
        
    out_file = "labeled_" + os.path.basename(file_path)
    with open(out_file, "w", encoding="utf-8") as f:
        f.writelines(output_lines)
    print(f"[Успех] Файл размечен и сохранен как '{out_file}'!")

# =====================================================================
# CLI ROUTER
# =====================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Пиратский NER-Инструмент v1.1")
        print("Использование:")
        print("  uv run pirate_ner.py train              - Скачать 15k строк и обучить граф")
        print("  uv run pirate_ner.py text 'Ваша фраза'  - Разметить одну фразу в консоли")
        print("  uv run pirate_ner.py file path/to.txt   - Разметить целый текстовый файл")
        sys.exit(0)
        
    command = sys.argv[1].lower()
    
    if command == "train":
        train_and_save()
    elif command == "text":
        if len(sys.argv) < 3:
            print("Ошибка: Передайте текст в кавычках.")
        else:
            process_text(sys.argv[2])
    elif command == "file":
        if len(sys.argv) < 3:
            print("Ошибка: Укажите путь к файлу.")
        else:
            process_file(sys.argv[2])
    else:
        print(f"Неизвестная команда '{command}'.")
