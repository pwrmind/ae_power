import networkx as nx

# 1. Обучающий текст (корпус)
corpus = [
    "я люблю программировать",
    "я люблю создавать легкие модели",
    "создавать модели это круто",
    "легкие модели работают быстро"
]

# 2. Строим граф токенов
# Узлы — это уникальные слова. Внутреннее значение V(y) — частота слова в тексте.
G = nx.Graph()
word_counts = {}

# Считаем частоту слов
for sentence in corpus:
    words = sentence.split()
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1

# Добавляем узлы со значениями V(y)
for word, count in word_counts.items():
    G.add_node(word, val=count)

# Связываем слова, которые стоят рядом в предложениях
for sentence in corpus:
    words = sentence.split()
    for i in range(len(words) - 1):
        G.add_edge(words[i], words[i+1])

# 3. Матрица крыльев (расстояние в шагах графа -> вес внимания)
# 1 шаг = 0.25, 2 шага = 0.0625, 3 шага = 0.015625
def get_wing_weight(distance):
    if distance == 1: return 0.25
    if distance == 2: return 0.0625
    if distance == 3: return 0.015625
    return 0.0

# 4. Функция предсказания следующего токена по вашему алгоритму
def predict_next_token(current_node, graph, visited_nodes):
    candidates = []
    
    # Считаем расстояния от текущего узла до ВСЕХ остальных в графе
    for node in graph.nodes:
        if node == current_node or node in visited_nodes:
            continue
            
        try:
            # Находим кратчайшее расстояние в шагах
            distance = nx.shortest_path_length(graph, source=current_node, target=node)
            wing_weight = get_wing_weight(distance)
            
            if wing_weight > 0:
                node_val = graph.nodes[node]['val']
                # Ваша формула: Вес крыла + Значение узла
                score = wing_weight + node_val
                candidates.append((node, score))
        except nx.NetworkXNoPath:
            continue  # Если узлы не связаны вообще
            
    if not candidates:
        return None
        
    # Сортируем по силе связи и берем строго Top-2 самых сильных связей
    top_2 = sorted(candidates, key=lambda x: x[1], reverse=True)[:2]
    
    print(f"\nФокус на слове '{current_node}'. Из 'всех со всеми' выбраны Top-2 связи:")
    for rank, (node, score) in enumerate(top_2, 1):
        print(f"  {rank}. Слово '{node}' -> Балл: {score:.4f}")
        
    # Выбираем финального победителя по максимальному значению
    winner = top_2[0][0]
    return winner

# --- ЗАПУСК ГЕНЕРАЦИИ ---
start_word = "я"
current = start_word
generated_text = [current]
visited = {current} # чтобы не зацикливаться на одном слове

print("--- Старт генерации текста ---")
for _ in range(4):
    next_word = predict_next_token(current, G, visited)
    if not next_word:
        print("\n[Конец графа или нет доступных связей]")
        break
    generated_text.append(next_word)
    visited.add(next_word)
    current = next_word

print("\nФинальный сгенерированный текст:")
print(" ".join(generated_text))
