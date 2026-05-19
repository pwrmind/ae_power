from datasets import load_dataset
import re

# =====================================================================
# 1. ЗАГРУЗКА ИСТИННОГО ДАТАСЕТА С HUGGING FACE ЧЕРЕЗ DATASETS
# =====================================================================
print("[Hugging Face Атака] Загружаем оригинальный датасет WikiAnn для русского языка...")

# Загружаем официальный датасет WikiAnn (конфигурация 'ru' - русский язык)
raw_dataset = load_dataset("wikiann", "ru", split="train[:1500]") 

print(f"[Успех] Из Hugging Face извлечено {len(raw_dataset)} реальных предложений Википедии!")

# Соответствие числовых ID тегов их строковым именам в WikiAnn
# 0: O, 1: B-PER, 2: I-PER, 3: B-ORG, 4: I-ORG, 5: B-LOC, 6: I-LOC
tags_features = raw_dataset.features["ner_tags"].feature.names
id_to_tag = {i: name for i, name in enumerate(tags_features)}

def get_suffix(word):
    return f"suff_{word[-3:]}" if len(word) >= 3 else None

# =====================================================================
# 2. ОБУЧЕНИЕ: Строим позиционно-вероятностный граф
# =====================================================================
token_counts = {}
tag_counts = {tag: 0 for tag in tags_features}
graph_left_context = {}  
graph_right_context = {} 
graph_suffix_to_tag = {} 

for item in raw_dataset:
    # Токены в датасете WikiAnn уже идеально нарезаны, просто приводим к нижнему регистру
    tokens = [t.lower() for t in item["tokens"]]
    # Переводим числовые ID тегов в понятные строки (например, 1 -> 'B-PER')
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
            
        # Позиционный радар (Окно внимания до 5 шагов в Лево/Право)
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

print("\n--- Обучение на официальных данных завершено! ---")
print(f"Изучено уникальных слов из Википедии: {len(token_counts)}")
print(f"Изучено морфологических суффиксов: {len(graph_suffix_to_tag)}")
print(f"Распределение тегов в реальном датасете: {tag_counts}")

# Геометрическая прогрессия крыльев внимания 1/4^x
def get_wing_weight(distance):
    weights = {0: 1.0, 1: 0.25, 2: 0.0625, 3: 0.015625, 4: 0.003906, 5: 0.000976}
    return weights.get(distance, 0.0)

# =====================================================================
# 3. АЛГОРИТМ ИНФЕРЕНСА С НОРМАЛИЗАЦИЕЙ ВЕРОЯТНОСТЕЙ
# =====================================================================
def predict_ner_huggingface(test_sentence):
    test_sentence = test_sentence.lower()
    tokens = re.findall(r'\w+|[^\w\s]', test_sentence)
    
    print(f"\n--- Тестирование на фразе: '{test_sentence}' ---")
    
    for target_idx, target_token in enumerate(tokens):
        # Базовые заряды Лапласа
        tag_scores = {tag: 0.0001 for tag in tag_counts}
        
        # Сила 1: Морфология (суффиксы реальной Википедии)
        suff = get_suffix(target_token)
        if suff and suff in graph_suffix_to_tag:
            for tag, count in graph_suffix_to_tag[suff].items():
                tag_scores[tag] += (count / tag_counts[tag]) * 1.5
        
        # Сила 2: Поля внимания (5 шагов)
        for context_idx, context_token in enumerate(tokens):
            if context_token == target_token: continue
                
            distance = abs(target_idx - context_idx)
            wing_weight = get_wing_weight(distance)
            
            if wing_weight > 0:
                if context_idx < target_idx: # Сосед слева
                    if context_token in graph_right_context and distance in graph_right_context[context_token]:
                        for tag, count in graph_right_context[context_token][distance].items():
                            tag_scores[tag] += wing_weight * (count / tag_counts[tag])
                            
                elif context_idx > target_idx: # Сосед справа
                    if context_token in graph_left_context and distance in graph_left_context[context_token]:
                        for tag, count in graph_left_context[context_token][distance].items():
                            tag_scores[tag] += wing_weight * (count / tag_counts[tag])

        # Выбираем самый заряженный тег
        best_tag = max(tag_scores, key=tag_scores.get)
        if all(v == 0.0001 for v in tag_scores.values()):
            best_tag = "O"
            
        clean_scores = {k: round(v, 4) for k, v in tag_scores.items() if v > 0.0001}
        print(f"Токен: '{target_token:<15}' -> Тег: {best_tag:<7} | Заряды полей: {clean_scores}")

# =====================================================================
# 4. СУРОВЫЕ ИСПЫТАНИЯ НА НЕЗНАКОМЫХ ДАННЫХ
# =====================================================================
# Эксперимент 1: Известные сущности в новой связке
predict_ner_huggingface("Александр Пушкин родился в Москве")

# Эксперимент 2: Новая сущность, которой точно нет в WikiAnn (Организация)
predict_ner_huggingface("Яндекс разрабатывает новые технологии")
