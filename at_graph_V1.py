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

# ИСПОЛЬЗУЕМ НАПРАВЛЕННЫЙ ГРАФ (DiGraph)
G = nx.DiGraph()

for ngram, count in ngram_counts.items():
    G.add_node(ngram, val=count)

# Связи теперь строго направленные: от текущего к следующему
for seq in all_ngrams_sequences:
    for i in range(len(seq) - 1):
        G.add_edge(seq[i], seq[i+1])

# Матрица крыльев внимания
def get_wing_weight(distance):
    if distance == 1: return 0.25
    if distance == 2: return 0.0625
    if distance == 3: return 0.015625
    return 0.0

# Алгоритм предсказания следующего токена
def predict_next_ngram(current_node, graph, visited_nodes):
    candidates = []
    
    for node in graph.nodes:
        if node == current_node or node in visited_nodes:
            continue
            
        try:
            # Для направленного графа считаем путь только ВПЕРЕД по стрелкам
            distance = nx.shortest_path_length(graph, source=current_node, target=node)
            wing_weight = get_wing_weight(distance)
            
            if wing_weight > 0:
                node_val = graph.nodes[node]['val']
                
                # ИСПРАВЛЕННАЯ ФОРМУЛА: УМНОЖЕНИЕ ВМЕСТО СЛОЖЕНИЯ
                score = wing_weight * node_val
                candidates.append((node, score))
        except nx.NetworkXNoPath:
            continue
            
    if not candidates:
        return None
        
    # Выбираем Top-2
    top_2 = sorted(candidates, key=lambda x: x[1], reverse=True)[:2]
    
    print(f"\nФокус на '{current_node}'. Top-2 связи:")
    for rank, (node, score) in enumerate(top_2, 1):
        print(f"  {rank}. '{node}' -> Балл: {score:.4f}")
        
    return top_2[0][0] # Финальный победитель

# --- ЗАПУСК ГЕНЕРАЦИИ ---
start_ngram = "я л"
current = start_ngram
generated_ngrams = [current]
visited = {current}

print("--- Старт улучшенной N-граммной генерации ---")
for _ in range(25): # Увеличим число шагов, чтобы дойти до конца
    next_ngram = predict_next_ngram(current, G, visited)
    if not next_ngram:
        print("\n[Достигнут тупик или конец графа]")
        break
    generated_ngrams.append(next_ngram)
    visited.add(next_ngram)
    current = next_ngram

# Корректная сборка текста из перекрывающихся N-грамм
final_text = generated_ngrams[0]
for ngram in generated_ngrams[1:]:
    final_text += ngram[-1]

print("\nСгенерированный текст:")
print(f"[{final_text}]")
