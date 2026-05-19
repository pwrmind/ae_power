import networkx as nx

# 1. Обучающий корпус
corpus = [
    "я люблю программировать",
    "я люблю создавать легкие модели",
    "создавать модели это круто",
    "легкие модели работают быстро"
]

# --- ДОРАБОТКА 1: Пробел (символ '_') теперь жестко привязан к началу слов ---
def strict_tokenize(text):
    words = text.split(" ")
    tokens = []
    
    replacements = {
        "я": ["я"],
        "люблю": ["_люб", "лю"],
        "программировать": ["_программ", "ировать"],
        "создавать": ["_созда", "вать"],
        "модели": ["_модел", "и"],
        "легкие": ["_легк", "ие"],
        "это": ["_это"],
        "круто": ["_крут", "о"],
        "работают": ["_работ", "ают"],
        "быстро": ["_быстр", "о"]
    }
    
    for i, word in enumerate(words):
        lookup = f"_{word}" if i > 0 else word
        if lookup in replacements:
            tokens.extend(replacements[lookup])
        else:
            tokens.append(lookup)
    return tokens

tokenized_corpus = [strict_tokenize(sentence) for sentence in corpus]

# Считаем частоту токенов
token_counts = {}
for seq in tokenized_corpus:
    for t in seq:
        token_counts[t] = token_counts.get(t, 0) + 1

# Строим строгий направленный граф знаний
G_knowledge = nx.DiGraph()
for token, count in token_counts.items():
    G_knowledge.add_node(token, val=count)

for seq in tokenized_corpus:
    for i in range(len(seq) - 1):
        G_knowledge.add_edge(seq[i], seq[i+1])

# Расширенные крылья фокуса внимания (геометрическая прогрессия 1/4^x)
def get_extended_wing_weight(distance):
    weights = {1: 0.25, 2: 0.0625, 3: 0.015625, 4: 0.003906, 5: 0.000976}
    return weights.get(distance, 0.0)

# Функция оценки кандидата на основе знаний и затухающей истории
def evaluate_candidate(node, current_node, G_static, G_history, visited_nodes):
    if node == current_node or node in visited_nodes:
        return 0.0
        
    try:
        dist_static = nx.shortest_path_length(G_static, source=current_node, target=node)
        wing_static = get_extended_wing_weight(dist_static)
        
        if wing_static > 0:
            node_val = G_static.nodes[node]['val']
            base_score = wing_static * node_val
            
            # --- ДОРАБОТКА 2: Затухание памяти (Memory Decay) ---
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

# --- ДОРАБОТКА 3: Исправленный Beam Search (сортировка по весам) ---
def beam_search_step(beams, G_static, max_beams=2):
    new_candidates = []
    
    for G_hist, text_seq, visited, cumulative_score in beams:
        current_node = text_seq[-1]
        
        # Состариваем узлы в текущем графе истории на этом луче
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

# --- ЗАПУСК ДВИЖКА ---
print("--- Старт исправленного движка (Связанные пробелы) ---")

G_start_hist = nx.DiGraph()
G_start_hist.add_node("я", age=1)
initial_beam = (G_start_hist, ["я"], {"я"}, 0.0)
beams = [initial_beam]

# Хранилище для финального лучшего луча на случай внезапного конца графа
last_valid_beam = initial_beam

for step in range(15):
    next_beams = beam_search_step(beams, G_knowledge, max_beams=2)
    if not next_beams:
        print("[Конец графа]")
        break
        
    beams = next_beams
    last_valid_beam = beams[0] # Запоминаем текущего лидера
    
    best_tokens = beams[0][1]
    best_score = beams[0][3]
    current_text = "".join(best_tokens).replace("_", " ")
    print(f"Шаг {step+1} | Путь: '{current_text}' (Балл: {best_score:.4f})")

# Извлекаем данные из последней сохраненной стабильной точки
final_history_graph, final_token_sequence, _, final_score = last_valid_beam

print("\n--- Финал чистой генерации ---")
final_text_clean = "".join(final_token_sequence).replace("_", " ")
print(f"Результат: [{final_text_clean}]")
