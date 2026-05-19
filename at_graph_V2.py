import networkx as nx

# 1. Корпус текстов
corpus = [
    "я люблю программировать",
    "я люблю создавать легкие модели",
    "создавать модели это круто",
    "легкие модели работают быстро"
]

def text_to_ngrams(text, n=3):
    return [text[i:i+n] for i in range(len(text) - n + 1)]

all_ngrams_sequences = [text_to_ngrams(sentence, n=3) for sentence in corpus]

ngram_counts = {}
for seq in all_ngrams_sequences:
    for ngram in seq:
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

# СТАТИЧЕСКИЙ ГРАФ ЗНАНИЙ
G_knowledge = nx.DiGraph()
for ngram, count in ngram_counts.items():
    G_knowledge.add_node(ngram, val=count)

for seq in all_ngrams_sequences:
    for i in range(len(seq) - 1):
        G_knowledge.add_edge(seq[i], seq[i+1])

# УВЕЛИЧЕННЫЕ КРЫЛЬЯ ВНИМАНИЯ (до 5 уровней)
def get_extended_wing_weight(distance):
    weights = {1: 0.25, 2: 0.0625, 3: 0.015625, 4: 0.003906, 5: 0.000976}
    return weights.get(distance, 0.0)

# АЛГОРИТМ ПРЕДСКАЗАНИЯ С ДВУМЯ ГРАФАМИ
def predict_next_with_history_graph(current_node, G_static, G_history, visited_nodes):
    candidates = []
    
    for node in G_static.nodes:
        if node == current_node or node in visited_nodes:
            continue
            
        try:
            # 1. Смотрим вперед по статическому графу знаний
            dist_static = nx.shortest_path_length(G_static, source=current_node, target=node)
            wing_static = get_extended_wing_weight(dist_static)
            
            if wing_static > 0:
                node_val = G_static.nodes[node]['val']
                base_score = wing_static * node_val
                
                # 2. НАЛОЖЕНИЕ НА ГРАФ ИСТОРИИ (Вектор Времени)
                # Проверяем, связан ли кандидат с недавними узлами в нашей истории
                history_bonus = 1.0
                if node in G_history.nodes:
                    try:
                        # Считаем расстояние назад по истории до этого узла
                        dist_hist = nx.shortest_path_length(G_history, source=node, target=current_node)
                        wing_hist = get_extended_wing_weight(dist_hist)
                        if wing_hist > 0:
                            # Чем ближе узел в истории, тем сильнее он модулирует выбор
                            history_bonus += wing_hist * 10 # Коэффициент усиления памяти
                    except nx.NetworkXNoPath:
                        pass
                
                final_score = base_score * history_bonus
                candidates.append((node, final_score))
                
        except nx.NetworkXNoPath:
            continue
            
    if not candidates:
        return None
        
    top_2 = sorted(candidates, key=lambda x: x[1], reverse=True)[:2]
    
    print(f"\nФокус на '{current_node}'. Top-2 (с учетом Графа Истории):")
    for rank, (node, score) in enumerate(top_2, 1):
        print(f"  {rank}. '{node}' -> Итоговый балл: {score:.4f}")
        
    return top_2[0][0]

# --- ЗАПУСК ГЕНЕРАЦИИ ---
# ДИНАМИЧЕСКИЙ ГРАФ ИСТОРИИ (изначально пустой)
G_history = nx.DiGraph()

start_ngram = "я л"
current = start_ngram
generated_ngrams = [current]
visited = {current}

G_history.add_node(current) # Добавляем точку старта в историю

print("--- Старт генерации: Большие Крылья + Граф Истории ---")
for _ in range(35): # Шагов теперь больше, благодаря памяти модель не должна улетать в хаос
    next_ngram = predict_next_with_history_graph(current, G_knowledge, G_history, visited)
    if not next_ngram:
        print("\n[Конец доступных путей]")
        break
        
    # Динамически обновляем граф истории: строим мост из настоящего в будущее
    G_history.add_node(next_ngram)
    G_history.add_edge(current, next_ngram)
    
    generated_ngrams.append(next_ngram)
    visited.add(next_ngram)
    current = next_ngram

# Декодирование
final_text = generated_ngrams[0]
for ngram in generated_ngrams[1:]:
    final_text += ngram[-1]

print("\nРезультат генерации с двухпотоковым графовым вниманием:")
print(f"[{final_text}]")
