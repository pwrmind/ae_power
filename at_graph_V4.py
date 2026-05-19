import re
from collections import Counter
import networkx as nx

# 1. РЕАЛЬНЫЕ ДАННЫЕ (Вы можете вставить сюда любой живой текст, например, логи или статьи)
raw_data = [
    "привет, как твои дела?",
    "привет! программировать это очень интересно и круто.",
    "как работает этот сверхлегкий механизм внимания?",
    "этот механизм работает быстро и эффективно на реальных данных.",
    "данных много не бывает, когда мы обучаем модели.",
    "модели создаются для решения сложных задач программирования.",
    "задачи бывают легкие, а бывают очень сложные.",
    "сверхлегкий алгоритм позволяет запускать модели на процессоре."
]

# --- ДОРАБОТКА 1: Автоматический BPE-like токенизатор для реального текста ---
def build_vocab_and_tokenize(texts, max_vocab_size=100):
    """
    Чистит текст, привязывает пробелы к началу слов (маркер '_')
    и разбивает длинные/редкие слова на подслова (subwords).
    """
    # Базовая очистка
    cleaned_texts = []
    for text in texts:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text) # убираем пунктуацию
        cleaned_texts.append(text)
    
    # Считаем частоту «сырых» слов со связанными пробелами
    raw_word_counts = Counter()
    for text in cleaned_texts:
        words = text.split()
        for i, word in enumerate(words):
            token = f"_{word}" if i > 0 else word
            raw_word_counts[token] += 1
            
    # Выделяем частые слова (они останутся целыми токенами)
    frequent_tokens = {word for word, count in raw_word_counts.items() if count >= 2}
    
    # Функция для нарезки слова на кусочки, если оно редкое
    def segment_word(word_token):
        if word_token in frequent_tokens or len(word_token) <= 4:
            return [word_token]
        # Если слово редкое и длинное, бьем его по 4 символа (имитация subwords)
        return [word_token[i:i+4] for i in range(0, len(word_token), 4)]

    # Токенизируем весь наш датасет
    tokenized_dataset = []
    for text in cleaned_texts:
        words = text.split()
        sentence_tokens = []
        for i, word in enumerate(words):
            word_token = f"_{word}" if i > 0 else word
            sentence_tokens.extend(segment_word(word_token))
        tokenized_dataset.append(sentence_tokens)
        
    return tokenized_dataset

# Запускаем токенизацию реальных данных
tokenized_corpus = build_vocab_and_tokenize(raw_data)

# Считаем частоту токенов для значений узлов V(y)
token_counts = Counter(t for seq in tokenized_corpus for t in seq)

# 2. СТРОИМ ГРАФ ЗНАНИЙ
G_knowledge = nx.DiGraph()
for token, count in token_counts.items():
    G_knowledge.add_node(token, val=count)

for seq in tokenized_corpus:
    for i in range(len(seq) - 1):
        G_knowledge.add_edge(seq[i], seq[i+1])

print(f"Размер графа знаний: {G_knowledge.number_of_nodes()} узлов, {G_knowledge.number_of_edges()} связей.")

# Расширенные крылья фокуса внимания (геометрическая прогрессия 1/4^x)
def get_extended_wing_weight(distance):
    weights = {1: 0.25, 2: 0.0625, 3: 0.015625, 4: 0.003906, 5: 0.000976}
    return weights.get(distance, 0.0)

# Функция оценки кандидата
def evaluate_candidate(node, current_node, G_static, G_history, visited_nodes):
    if node == current_node or node in visited_nodes:
        return 0.0
        
    try:
        dist_static = nx.shortest_path_length(G_static, source=current_node, target=node)
        wing_static = get_extended_wing_weight(dist_static)
        
        if wing_static > 0:
            node_val = G_static.nodes[node]['val']
            base_score = wing_static * node_val
            
            # Затухание памяти в истории
            history_bonus = 1.0
            if node in G_history.nodes:
                try:
                    dist_hist = nx.shortest_path_length(G_history, source=node, target=current_node)
                    wing_hist = get_extended_wing_weight(dist_hist)
                    if wing_hist > 0:
                        age = G_history.nodes[node].get('age', 1)
                        decay_factor = 1.0 / (age ** 0.5)
                        history_bonus += (wing_hist * 10) * decay_factor
                except nx.NetworkXNoPath:
                    pass
                    
            return base_score * history_bonus
    except nx.NetworkXNoPath:
        pass
    return 0.0

# Beam Search
def beam_search_step(beams, G_static, max_beams=3):
    new_candidates = []
    
    for G_hist, text_seq, visited, cumulative_score in beams:
        current_node = text_seq[-1]
        
        for node in G_hist.nodes:
            G_hist.nodes[node]['age'] = G_hist.nodes[node].get('age', 0) + 1
            
        for node in G_static.nodes:
            score = evaluate_candidate(node, current_node, G_static, G_hist, visited)
            if score > 0:
                G_hist_clone = G_hist.copy()
                G_hist_clone.add_node(node, age=1)
                G_hist_clone.add_edge(current_node, node)
                
                new_candidates.append((
                    G_hist_clone,
                    text_seq + [node],
                    visited | {node},
                    cumulative_score + score
                ))
                
    if not new_candidates:
        return []
        
    return sorted(new_candidates, key=lambda x: x[3], reverse=True)[:max_beams]

# --- ЗАПУСК НА РЕАЛЬНЫХ ДАННЫХ ---
print("\n--- Тестирование на реальном корпусе ---")

# Зададим стартовый токен из нашей базы (например, "этот")
start_token = "этот"

if start_token not in G_knowledge:
    print(f"Ошибка: Стартовый токен '{start_token}' отсутствует в графе знаний.")
    exit()

G_start_hist = nx.DiGraph()
G_start_hist.add_node(start_token, age=1)
initial_beam = (G_start_hist, [start_token], {start_token}, 0.0)
beams = [initial_beam]
last_valid_beam = initial_beam

# Генерируем до 15 шагов
for step in range(15):
    next_beams = beam_search_step(beams, G_knowledge, max_beams=3)
    if not next_beams:
        print("[Конец графа (достигнут тупик без продолжений)]")
        break
        
    beams = next_beams
    last_valid_beam = beams[0]
    
    best_tokens = beams[0][1]
    best_score = beams[0][3]
    current_text = "".join(best_tokens).replace("_", " ")
    print(f"Шаг {step+1} | '{current_text}' (Score: {best_score:.4f})")

# Итоговый вывод
_, final_token_sequence, _, _ = last_valid_beam
final_text_clean = "".join(final_token_sequence).replace("_", " ")
print("\n--- Финал чистой генерации ---")
print(f"Входной токен:   [{start_token}]")
print(f"Результат модели: [{final_text_clean}]")
