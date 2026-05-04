import json
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional
import itertools
from scipy import sparse
from scipy.stats import poisson
import warnings

warnings.filterwarnings('ignore')


class AuthorCIDRE:
    """
    Адаптированная версия CIDRE для обнаружения аномальных групп авторов,
    чрезмерно цитирующих друг друга.
    """

    def __init__(self, data_file: str = "openalex_data.json"):
        """
        Инициализация с загрузкой данных из JSON файла.
        """
        print("Загрузка данных...")
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # Проверяем структуру данных
        print("Ключи в данных:", self.data.keys())

        # Проверяем разные возможные структуры
        if 'results' in self.data:
            self.works = self.data.get("results", [])
            print(f"Найдена структура с 'results': {len(self.works)} работ")
        elif 'works' in self.data:
            self.works = self.data.get("works", [])
            print(f"Найдена структура с 'works': {len(self.works)} работ")
        elif isinstance(self.data, list):
            self.works = self.data
            print(f"Данные представляют собой список: {len(self.works)} работ")
        else:
            # Пробуем найти работы в других ключах
            for key in self.data.keys():
                if isinstance(self.data[key], list) and len(self.data[key]) > 0:
                    if 'title' in self.data[key][0] or 'id' in self.data[key][0]:
                        self.works = self.data[key]
                        print(f"Найдены работы в ключе '{key}': {len(self.works)} работ")
                        break
            else:
                self.works = []
                print("Не удалось найти список работ")

        self.metadata = self.data.get("metadata", {})

        print(f"Всего работ: {len(self.works)}")

        # Показываем структуру первой работы для отладки
        if self.works:
            print("\nСтруктура первой работы:")
            print(f"Ключи: {list(self.works[0].keys())}")

            # Проверяем наличие referenced_works
            if 'referenced_works' in self.works[0]:
                print(f"Есть referenced_works: {len(self.works[0]['referenced_works'])} ссылок")
            else:
                print("Нет referenced_works в первой работе")

            # Проверяем наличие authorships
            if 'authorships' in self.works[0]:
                print(f"Есть authorships: {len(self.works[0]['authorships'])} авторов")
            else:
                print("Нет authorships в первой работе")

        # Создаем индекс работ по ID для быстрого поиска
        self.works_by_id = {}
        for work in self.works:
            work_id = work.get("id")
            if work_id:
                self.works_by_id[work_id] = work
            else:
                # Пробуем другие возможные ключи для ID
                if "ids" in work and isinstance(work["ids"], dict):
                    for id_type in ["openalex", "doi", "mag"]:
                        if id_type in work["ids"]:
                            self.works_by_id[work["ids"][id_type]] = work
                            break

        print(f"Создан индекс для {len(self.works_by_id)} работ")

        # Основные структуры данных
        self.author_graph = None  # Граф цитирований между авторами
        self.author_to_id = {}  # Имя автора -> числовой ID
        self.id_to_author = {}  # числовой ID -> автор
        self.author_blocks = {}  # Блоки (сообщества) авторов
        self.excessive_edges = set()  # Избыточные цитаты

    def build_author_citation_graph(self, min_year: int = None, max_year: int = None):
        """
        Построение графа цитирований между авторами.

        Args:
            min_year, max_year: Фильтр по годам публикации
        """
        print("\n" + "=" * 60)
        print("Построение графа цитирований между авторами...")

        if not self.works:
            print("Нет данных о работах!")
            return np.zeros((0, 0))

        # Сначала соберем всех уникальных авторов
        all_authors = set()
        works_with_authors = 0

        for work in self.works:
            if min_year and work.get("publication_year", 0) < min_year:
                continue
            if max_year and work.get("publication_year", 0) > max_year:
                continue

            authorships = work.get("authorships", [])
            if not authorships:
                continue

            works_with_authors += 1
            for authorship in authorships:
                author_info = authorship.get("author", {})
                if author_info and "id" in author_info:
                    author_id = author_info["id"]
                    if author_id is not None and author_id != "":  # Ключевое исправление
                        all_authors.add(author_id)
                else:
                    # Пробуем получить ID из других мест
                    raw_name = authorship.get("raw_author_name", "")
                    if raw_name:
                        all_authors.add(f"raw_{raw_name}")

        print(f"Найдено {len(all_authors)} уникальных авторов в {works_with_authors} работах")

        if not all_authors:
            print("Нет данных об авторах!")
            return np.zeros((0, 0))

        # Удаляем None значения
        all_authors = {author for author in all_authors if author is not None}

        # Создаем маппинг авторов на числовые ID
        # Фильтруем None перед сортировкой
        valid_authors = [author for author in all_authors if author is not None]
        valid_authors.sort()

        self.author_to_id = {author: idx for idx, author in enumerate(valid_authors)}
        self.id_to_author = {idx: author for author, idx in self.author_to_id.items()}

        n_authors = len(valid_authors)
        print(f"Количество авторов для анализа: {n_authors}")

        # Инициализируем матрицу смежности
        author_citations = np.zeros((n_authors, n_authors), dtype=int)

        # Собираем статистику для каждой работы
        citation_count = 0
        works_processed = 0
        works_with_references = 0

        # Сначала создадим список всех работ с их авторами для быстрого поиска
        work_authors_cache = {}
        for work in self.works:
            work_id = work.get("id", "")
            if not work_id:
                continue

            authors = []
            for authorship in work.get("authorships", []):
                author_info = authorship.get("author", {})
                if author_info and "id" in author_info:
                    author_id = author_info["id"]
                    if author_id in self.author_to_id:
                        authors.append(self.author_to_id[author_id])
                else:
                    raw_name = authorship.get("raw_author_name", "")
                    if raw_name:
                        author_key = f"raw_{raw_name}"
                        if author_key in self.author_to_id:
                            authors.append(self.author_to_id[author_key])

            work_authors_cache[work_id] = authors

        print(f"Создан кэш авторов для {len(work_authors_cache)} работ")

        # Теперь обрабатываем цитирования
        for work in self.works:
            if min_year and work.get("publication_year", 0) < min_year:
                continue
            if max_year and work.get("publication_year", 0) > max_year:
                continue

            works_processed += 1

            # Авторы цитирующей статьи
            citing_work_id = work.get("id", "")
            citing_authors = work_authors_cache.get(citing_work_id, [])

            if not citing_authors:
                continue

            # Цитируемые работы
            referenced_works = work.get("referenced_works", [])
            if not referenced_works:
                continue

            works_with_references += 1

            for ref_id in referenced_works:
                # Ищем авторов цитируемой работы
                cited_authors = []

                # Пробуем разные форматы ID
                possible_ref_ids = [ref_id]
                if ref_id.startswith("https://"):
                    possible_ref_ids.append(ref_id.replace("https://", ""))

                for possible_id in possible_ref_ids:
                    if possible_id in work_authors_cache:
                        cited_authors = work_authors_cache[possible_id]
                        break
                    elif possible_id in self.works_by_id:
                        # Если работа в индексе, но нет в кэше авторов
                        cited_work = self.works_by_id[possible_id]
                        authors = []
                        for authorship in cited_work.get("authorships", []):
                            author_info = authorship.get("author", {})
                            if author_info and "id" in author_info:
                                author_id = author_info["id"]
                                if author_id in self.author_to_id:
                                    authors.append(self.author_to_id[author_id])
                            else:
                                raw_name = authorship.get("raw_author_name", "")
                                if raw_name:
                                    author_key = f"raw_{raw_name}"
                                    if author_key in self.author_to_id:
                                        authors.append(self.author_to_id[author_key])

                        work_authors_cache[possible_id] = authors
                        cited_authors = authors
                        break

                if not cited_authors:
                    continue

                # Добавляем цитирования между всеми авторами
                for citing in citing_authors:
                    for cited in cited_authors:
                        if citing != cited:  # Исключаем самоцитирования
                            author_citations[citing, cited] += 1
                            citation_count += 1

            if works_processed % 100 == 0 and works_processed > 0:
                print(f"Обработано {works_processed}/{len(self.works)} работ...")

        print(f"Работ с ссылками: {works_with_references}")
        print(f"Всего цитирований между авторами: {citation_count}")

        if citation_count == 0:
            print("Предупреждение: не найдено цитирований между авторами!")
            return author_citations

        # Создаем граф NetworkX
        self.author_graph = nx.DiGraph()

        for i in range(n_authors):
            author_id = self.id_to_author[i]
            author_name = self._get_author_name_by_id(author_id)
            self.author_graph.add_node(i, author_id=author_id, name=author_name)

        # Добавляем ребра с весом > 0
        edges_added = 0
        for i in range(n_authors):
            for j in range(n_authors):
                if author_citations[i, j] > 0:
                    self.author_graph.add_edge(i, j, weight=author_citations[i, j])
                    edges_added += 1

        print(f"Граф построен: {self.author_graph.number_of_nodes()} узлов, "
              f"{edges_added} ребер")

        # Выводим статистику графа
        if edges_added > 0:
            degrees = [d for n, d in self.author_graph.degree(weight='weight')]
            print(f"Средняя степень: {np.mean(degrees):.2f}")
            print(f"Максимальная степень: {np.max(degrees)}")
            print(f"Минимальная степень: {np.min(degrees)}")

        return author_citations

    def _get_author_name_by_id(self, author_id: str) -> str:
        """Получение имени автора по его ID."""
        if author_id.startswith("raw_"):
            return author_id[4:]  # Убираем префикс "raw_"

        for work in self.works:
            for authorship in work.get("authorships", []):
                author_info = authorship.get("author", {})
                if author_info.get("id") == author_id:
                    return author_info.get("display_name", "Неизвестно")

        return "Неизвестно"

    def detect_author_communities(self, method: str = "connected_components"):
        """
        Обнаружение сообществ авторов для нулевой модели.
        """
        print("\n" + "=" * 60)
        print("Обнаружение сообществ авторов...")

        if not self.author_graph or self.author_graph.number_of_nodes() == 0:
            print("Граф пустой или не построен! Использую простую кластеризацию...")
            self._simple_community_detection()
            return self.author_blocks

        print(f"Граф имеет {self.author_graph.number_of_nodes()} узлов и "
              f"{self.author_graph.number_of_edges()} ребер")

        try:
            if method == "louvain":
                import community as community_louvain
                # Создаем неориентированный граф для Лувена
                undirected_graph = self.author_graph.to_undirected()
                partition = community_louvain.best_partition(undirected_graph)

                # Сохраняем блоки
                self.author_blocks = partition

                n_communities = len(set(partition.values()))
                print(f"Найдено {n_communities} сообществ методом Лувена")

            elif method == "connected_components":
                # Используем слабо связные компоненты как простой метод
                undirected_graph = self.author_graph.to_undirected()
                components = list(nx.connected_components(undirected_graph))

                self.author_blocks = {}
                for comp_id, component in enumerate(components):
                    for node in component:
                        self.author_blocks[node] = comp_id

                print(f"Найдено {len(components)} связных компонент")

            else:
                # Простой метод по умолчанию
                self._simple_community_detection()

            # Статистика по размерам сообществ
            if self.author_blocks:
                block_sizes = Counter(self.author_blocks.values())
                sizes = list(block_sizes.values())
                print(f"Размеры сообществ: от {min(sizes)} до {max(sizes)} авторов")
                print(f"Средний размер: {np.mean(sizes):.1f} авторов")

        except Exception as e:
            print(f"Ошибка при кластеризации: {e}")
            print("Использую простую кластеризацию...")
            self._simple_community_detection()

        return self.author_blocks

    def _simple_community_detection(self):
        """Простой эвристический метод кластеризации."""
        print("Используется простой метод кластеризации...")

        if not self.author_graph:
            print("Граф не построен, создаю тривиальные блоки...")
            n_authors = len(self.author_to_id)
            self.author_blocks = {i: i % 10 for i in range(n_authors)}
            print(f"Создано 10 тривиальных блоков")
            return

        # Группируем авторов по соавторству (более реалистично)
        coauthor_graph = nx.Graph()

        # Добавляем всех авторов
        for i in range(len(self.author_to_id)):
            coauthor_graph.add_node(i)

        # Строим граф соавторства на основе общих работ
        author_works = defaultdict(set)
        for work in self.works:
            work_authors = []
            for authorship in work.get("authorships", []):
                author_info = authorship.get("author", {})
                if author_info and "id" in author_info:
                    author_id = author_info["id"]
                    if author_id in self.author_to_id:
                        author_idx = self.author_to_id[author_id]
                        work_authors.append(author_idx)
                        author_works[author_idx].add(work.get("id", ""))

            # Добавляем связи между соавторами
            for i in range(len(work_authors)):
                for j in range(i + 1, len(work_authors)):
                    coauthor_graph.add_edge(work_authors[i], work_authors[j])

        # Используем связные компоненты графа соавторства
        components = list(nx.connected_components(coauthor_graph))

        self.author_blocks = {}
        for comp_id, component in enumerate(components):
            for node in component:
                self.author_blocks[node] = comp_id

        # Для авторов без соавторов создаем отдельные блоки
        next_block_id = len(components)
        for author_idx in range(len(self.author_to_id)):
            if author_idx not in self.author_blocks:
                self.author_blocks[author_idx] = next_block_id
                next_block_id += 1

        print(f"Создано {len(set(self.author_blocks.values()))} блоков на основе соавторства")

    def fit_dcSBM(self):
        """
        Применение degree-corrected Stochastic Block Model
        для расчета ожидаемого количества цитирований.
        """
        print("\n" + "=" * 60)
        print("Расчет ожидаемых цитирований по dcSBM...")

        if not self.author_graph or self.author_graph.number_of_nodes() == 0:
            print("Граф пустой, пропускаем dcSBM")
            return np.zeros((0, 0))

        if not self.author_blocks:
            self.detect_author_communities()

        n_authors = len(self.author_to_id)

        # Вычисляем силы узлов (исходящие и входящие)
        out_strength = np.zeros(n_authors)
        in_strength = np.zeros(n_authors)

        for i in range(n_authors):
            if i in self.author_graph:
                out_strength[i] = sum(self.author_graph[i][j]['weight']
                                      for j in self.author_graph.successors(i))
                in_strength[i] = sum(self.author_graph[j][i]['weight']
                                     for j in self.author_graph.predecessors(i))

        # Группируем авторов по блокам
        blocks = defaultdict(list)
        for author_idx, block_id in self.author_blocks.items():
            blocks[block_id].append(author_idx)

        n_blocks = len(blocks)
        print(f"Количество блоков в модели: {n_blocks}")

        if n_blocks == 0:
            print("Нет блоков для моделирования")
            return np.zeros((n_authors, n_authors))

        # Вычисляем параметры для каждого блока
        block_out_strength = defaultdict(float)
        block_in_strength = defaultdict(float)
        block_edges = defaultdict(lambda: defaultdict(float))

        # Сначала собираем статистику по рёбрам между блоками
        for i in range(n_authors):
            if i not in self.author_graph:
                continue

            block_i = self.author_blocks.get(i, -1)
            for j in self.author_graph.successors(i):
                block_j = self.author_blocks.get(j, -1)
                weight = self.author_graph[i][j]['weight']
                block_edges[block_i][block_j] += weight

        # Вычисляем суммы сил по блокам
        for author_idx in range(n_authors):
            block_id = self.author_blocks.get(author_idx, -1)
            block_out_strength[block_id] += out_strength[author_idx]
            block_in_strength[block_id] += in_strength[author_idx]

        # Вычисляем ожидаемые значения λ_ij
        self.expected_citations = np.zeros((n_authors, n_authors))

        for i in range(n_authors):
            block_i = self.author_blocks.get(i, -1)
            if block_i == -1 or block_out_strength[block_i] == 0:
                continue

            for j in range(n_authors):
                block_j = self.author_blocks.get(j, -1)
                if block_j == -1 or block_in_strength[block_j] == 0:
                    continue

                # Формула из статьи: λ_ij = s_out_i * s_in_j * Λ_uv / (S_out_u * S_in_v)
                if block_i >= 0 and block_j >= 0:
                    lambda_ij = (out_strength[i] * in_strength[j] *
                                 block_edges[block_i][block_j] /
                                 (block_out_strength[block_i] * block_in_strength[block_j]))
                else:
                    lambda_ij = 0

                # Клиппинг как в статье
                self.expected_citations[i, j] = max(1, lambda_ij)

        print("Модель dcSBM обучена")
        return self.expected_citations

    def find_excessive_citations(self, alpha: float = 0.01):
        """
        Нахождение избыточных цитирований с поправкой Бенджамини-Хохберга.
        """
        print("\n" + "=" * 60)
        print("Поиск избыточных цитирований...")

        if not hasattr(self, 'expected_citations'):
            print("Сначала обучаем dcSBM...")
            self.fit_dcSBM()

        if not self.author_graph or self.author_graph.number_of_nodes() == 0:
            print("Граф пустой, нет цитирований для анализа")
            self.excessive_edges = set()
            return set()

        n_authors = len(self.author_to_id)
        p_values = []
        edges_info = []

        # Собираем все p-значения
        edges_processed = 0
        for i in range(n_authors):
            if i not in self.author_graph:
                continue

            for j in self.author_graph.successors(i):
                if i == j:  # Пропускаем самоцитирования
                    continue

                actual = self.author_graph[i][j]['weight']
                expected = self.expected_citations[i, j]

                # p-value из распределения Пуассона
                if expected > 0:
                    # P(X >= actual) = 1 - P(X < actual) = 1 - sum_{k=0}^{actual-1} P(X=k)
                    p_val = 1 - poisson.cdf(actual - 1, expected)
                else:
                    p_val = 1.0

                p_values.append(p_val)
                edges_info.append((i, j, actual, expected, p_val))
                edges_processed += 1

        print(f"Обработано {edges_processed} рёбер")

        if not p_values:
            print("Нет рёбер для анализа")
            self.excessive_edges = set()
            return set()

        # Сортируем p-значения для поправки Бенджамини-Хохберга
        m = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = np.array(p_values)[sorted_indices]

        # Применяем поправку
        significant_edges = set()
        for idx, (rank, p_val) in enumerate(zip(range(1, m + 1), sorted_p_values)):
            threshold = (rank * alpha) / m
            if p_val <= threshold:
                # Находим соответствующее ребро
                edge_idx = sorted_indices[idx]
                i, j, _, _, _ = edges_info[edge_idx]
                significant_edges.add((i, j))
            else:
                break

        self.excessive_edges = significant_edges
        print(f"Найдено {len(self.excessive_edges)} избыточных цитирований")

        # Показываем примеры избыточных цитирований
        if significant_edges:
            print("\nПримеры избыточных цитирований:")
            for i, j in list(significant_edges)[:5]:
                actual = self.author_graph[i][j]['weight']
                expected = self.expected_citations[i, j]
                author_i = self._get_author_name(i)
                author_j = self._get_author_name(j)
                print(f"  {author_i} → {author_j}: {actual} (ожидалось: {expected:.2f}, "
                      f"отношение: {actual / expected:.1f}x)")

        return significant_edges

    def calculate_scores(self, author_group: Set[int]):
        """
        Вычисление donor и recipient scores для группы авторов.
        """
        if not self.excessive_edges:
            self.find_excessive_citations()

        scores = {}

        for author_idx in author_group:
            if author_idx not in self.author_graph:
                continue

            # Вычисляем исходящую и входящую силу
            out_strength = sum(self.author_graph[author_idx][j]['weight']
                               for j in self.author_graph.successors(author_idx))
            in_strength = sum(self.author_graph[j][author_idx]['weight']
                              for j in self.author_graph.predecessors(author_idx))

            if out_strength == 0 and in_strength == 0:
                continue

            # Donor score: доля избыточных цитирований, отправленных в группу
            excessive_out = 0
            for j in self.author_graph.successors(author_idx):
                if j in author_group and (author_idx, j) in self.excessive_edges:
                    excessive_out += self.author_graph[author_idx][j]['weight']

            donor_score = excessive_out / out_strength if out_strength > 0 else 0

            # Recipient score: доля избыточных цитирований, полученных от группы
            excessive_in = 0
            for j in self.author_graph.predecessors(author_idx):
                if j in author_group and (j, author_idx) in self.excessive_edges:
                    excessive_in += self.author_graph[j][author_idx]['weight']

            recipient_score = excessive_in / in_strength if in_strength > 0 else 0

            scores[author_idx] = {
                'donor_score': donor_score,
                'recipient_score': recipient_score,
                'out_strength': out_strength,
                'in_strength': in_strength
            }

        return scores

    def detect_anomalous_groups(self, theta: float = 0.15, min_citations: int = 10):
        """
        Основной алгоритм обнаружения аномальных групп.

        Args:
            theta: Минимальная доля избыточных цитирований
            min_citations: Минимальное количество цитирований внутри группы
        """
        print("\n" + "=" * 60)
        print(f"Поиск аномальных групп (θ={theta}, min_citations={min_citations})...")

        if not self.author_graph or self.author_graph.number_of_nodes() == 0:
            print("Граф пустой, нет данных для анализа")
            return []

        if not self.excessive_edges:
            self.find_excessive_citations()

        n_authors = len(self.author_to_id)

        # Шаг 1: Создаем подграф только с избыточными цитатами
        excessive_graph = nx.DiGraph()
        for i in range(n_authors):
            if i in self.author_graph:
                excessive_graph.add_node(i)

        for i, j in self.excessive_edges:
            if i in self.author_graph and j in self.author_graph:
                weight = self.author_graph[i][j]['weight']
                excessive_graph.add_edge(i, j, weight=weight)

        print(f"Подграф избыточных цитирований: {excessive_graph.number_of_nodes()} узлов, "
              f"{excessive_graph.number_of_edges()} рёбер")

        if excessive_graph.number_of_nodes() == 0:
            print("Нет избыточных цитирований для анализа")
            return []

        # Шаг 2: Итеративное удаление узлов (алгоритм k-core)
        U = set(excessive_graph.nodes())
        changed = True

        iteration = 0
        while changed and U:
            iteration += 1
            changed = False
            nodes_to_remove = set()

            scores = self.calculate_scores(U)

            for author_idx in U:
                if author_idx in scores:
                    donor_score = scores[author_idx]['donor_score']
                    recipient_score = scores[author_idx]['recipient_score']

                    if donor_score < theta and recipient_score < theta:
                        nodes_to_remove.add(author_idx)
                        changed = True
                else:
                    # Если нет данных для автора, удаляем его
                    nodes_to_remove.add(author_idx)
                    changed = True

            if nodes_to_remove:
                U -= nodes_to_remove
                print(f"Итерация {iteration}: удалено {len(nodes_to_remove)} авторов, "
                      f"осталось {len(U)}")

        if not U:
            print("После фильтрации не осталось авторов")
            return []

        # Шаг 3: Разделение на слабо связные компоненты
        subgraph = excessive_graph.subgraph(U)
        weak_components = list(nx.weakly_connected_components(subgraph))

        print(f"Найдено {len(weak_components)} слабо связных компонент")

        # Шаг 4: Фильтрация по минимальному количеству цитирований
        anomalous_groups = []

        for component in weak_components:
            # Считаем общее количество цитирований внутри группы
            total_citations = 0
            for i in component:
                for j in component:
                    if i != j and self.author_graph.has_edge(i, j):
                        total_citations += self.author_graph[i][j]['weight']

            if total_citations >= min_citations:
                excessive_count = self._count_excessive_in_group(component)
                anomalous_groups.append({
                    'authors': component,
                    'size': len(component),
                    'total_citations': total_citations,
                    'excessive_citations': excessive_count,
                    'excessive_ratio': excessive_count / total_citations if total_citations > 0 else 0
                })

        # Сортируем группы по количеству избыточных цитирований
        anomalous_groups.sort(key=lambda x: x['excessive_citations'], reverse=True)

        print(f"Найдено {len(anomalous_groups)} аномальных групп")

        return anomalous_groups

    def _count_excessive_in_group(self, group: Set[int]) -> int:
        """Подсчет избыточных цитирований внутри группы."""
        count = 0
        for i, j in self.excessive_edges:
            if i in group and j in group:
                count += self.author_graph[i][j]['weight']
        return count

    def analyze_groups(self, groups: List[Dict], top_k: int = 10):
        """
        Анализ и классификация найденных групп.
        """
        print("\n" + "=" * 60)
        print(f"Анализ {min(top_k, len(groups))} наиболее значимых групп:")

        if not groups:
            print("Нет групп для анализа")
            return

        for idx, group_info in enumerate(groups[:top_k]):
            authors = group_info['authors']
            print(f"\n{'=' * 60}")
            print(f"Группа #{idx + 1}")
            print(f"Размер: {group_info['size']} авторов")
            print(f"Всего цитирований внутри группы: {group_info['total_citations']}")
            print(f"Избыточные цитирования: {group_info['excessive_citations']}")
            print(f"Доля избыточных цитирований: {group_info['excessive_ratio']:.1%}")

            # Выводим информацию об авторах
            print("\nАвторы в группе:")
            for author_idx in sorted(authors):
                author_name = self._get_author_name(author_idx)

                # Статистика автора
                out_deg = self.author_graph.out_degree(author_idx, weight='weight')
                in_deg = self.author_graph.in_degree(author_idx, weight='weight')

                print(f"  - {author_name}")
                print(f"    Цитирует других: {out_deg}, "
                      f"Цитируется другими: {in_deg}")

            # Анализ паттернов цитирования
            self._analyze_citation_patterns(authors)

    def _analyze_citation_patterns(self, authors: Set[int]):
        """Анализ паттернов цитирования внутри группы."""
        print("\nАнализ паттернов цитирования:")

        # Проверяем концентрацию цитирований
        citation_counts = {}
        for i in authors:
            for j in authors:
                if i != j and self.author_graph.has_edge(i, j):
                    citation_counts[(i, j)] = self.author_graph[i][j]['weight']

        if not citation_counts:
            print("  Нет цитирований внутри группы")
            return

        total_citations = sum(citation_counts.values())

        # Самые частые связи
        top_pairs = sorted(citation_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        print("  Самые частые связи внутри группы:")
        for (i, j), count in top_pairs:
            author_i = self._get_author_name(i)
            author_j = self._get_author_name(j)
            percentage = (count / total_citations * 100) if total_citations > 0 else 0
            print(f"    {author_i} → {author_j}: {count} цитирований ({percentage:.1f}%)")

        # Проверяем взаимность
        reciprocal = 0
        reciprocal_pairs = []
        for i in authors:
            for j in authors:
                if i < j and self.author_graph.has_edge(i, j) and self.author_graph.has_edge(j, i):
                    reciprocal += 1
                    reciprocal_pairs.append((i, j))

        print(f"  Взаимных связей: {reciprocal}")
        if reciprocal_pairs:
            print("  Пары с взаимными цитированиями:")
            for i, j in reciprocal_pairs[:3]:  # Показываем первые 3 пары
                author_i = self._get_author_name(i)
                author_j = self._get_author_name(j)
                print(f"    {author_i} ↔ {author_j}")

    def _get_author_name(self, author_idx: int) -> str:
        """Получение имени автора по его индексу."""
        if author_idx in self.author_graph.nodes:
            return self.author_graph.nodes[author_idx].get('name', f"Автор_{author_idx}")
        return f"Автор_{author_idx}"

    def save_results(self, groups: List[Dict], output_file: str = "author_cartels.json"):
        """Сохранение результатов в JSON файл."""
        if not groups:
            print("Нет результатов для сохранения")
            return

        # Преобразуем numpy типы в стандартные Python типы
        total_excessive = sum(int(g['excessive_citations']) for g in groups)

        results = {
            'metadata': {
                'algorithm': 'AuthorCIDRE',
                'total_authors': int(len(self.author_to_id)),
                'total_works': int(len(self.works)),
                'theta': 0.15,
                'min_citations': 5,
                'groups_found': int(len(groups)),
                'total_excessive_citations': int(total_excessive)
            },
            'groups': []
        }

        for group_info in groups:
            group_data = {
                'size': int(group_info['size']),
                'total_citations': int(group_info['total_citations']),
                'excessive_citations': int(group_info['excessive_citations']),
                'excessive_ratio': float(group_info['excessive_ratio']),
                'authors': []
            }

            for author_idx in group_info['authors']:
                author_id = self.id_to_author[author_idx]
                author_name = self._get_author_name(author_idx)

                # Находим работы автора
                author_works = []
                for work in self.works:
                    for authorship in work.get("authorships", []):
                        author_info = authorship.get("author", {})
                        if author_info.get("id") == author_id:
                            author_works.append({
                                'work_id': work.get('id', ''),
                                'title': work.get('title', 'Без названия'),
                                'year': str(work.get('publication_year', '')),
                                'journal': work.get('primary_location', {}).get('source', {}).get('display_name', '')
                            })
                            break

                group_data['authors'].append({
                    'id': str(author_id),
                    'name': str(author_name),
                    'works_count': int(len(author_works)),
                    'sample_works': author_works[:3]  # Первые 3 работы
                })

            results['groups'].append(group_data)

        # Используем кастомный encoder для преобразования типов
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                return super().default(obj)

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

            print(f"\nРезультаты сохранены в {output_file}")
        except Exception as e:
            print(f"Ошибка при сохранении результатов: {e}")
            # Пробуем альтернативный метод сохранения
            #self._save_results_simple(groups, output_file)


def debug_data_structure(data_file: str = "openalex_data.json"):
    """Функция для отладки структуры данных."""
    print("=" * 60)
    print("ОТЛАДКА СТРУКТУРЫ ДАННЫХ")
    print("=" * 60)

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Тип данных: {type(data)}")
    print(f"Ключи в данных: {list(data.keys())}")

    # Проверяем разные возможные структуры
    for key in ['results', 'works', 'data', 'items']:
        if key in data and isinstance(data[key], list):
            print(f"\nНайден ключ '{key}' с {len(data[key])} элементами")
            if data[key]:
                first_item = data[key][0]
                print(f"Ключи первого элемента: {list(first_item.keys())[:20]}...")

                # Проверяем важные поля
                important_fields = ['id', 'title', 'authorships', 'referenced_works']
                for field in important_fields:
                    if field in first_item:
                        value = first_item[field]
                        if isinstance(value, list):
                            print(f"  {field}: список из {len(value)} элементов")
                        else:
                            print(f"  {field}: {type(value)}")
                    else:
                        print(f"  {field}: ОТСУТСТВУЕТ")

    # Если данные представляют собой список
    if isinstance(data, list):
        print(f"\nДанные представляют собой список из {len(data)} элементов")
        if data:
            print(f"Ключи первого элемента: {list(data[0].keys())[:20]}...")


def main():
    """Основная функция для запуска анализа."""

    # Сначала отладим структуру данных
    debug_data_structure("openalex_data.json")

    print("\n" + "=" * 60)
    print("ЗАПУСК АНАЛИЗА")
    print("=" * 60)

    # 1. Инициализация
    analyzer = AuthorCIDRE("openalex_data.json")

    # 2. Построение графа (можно ограничить годы)
    analyzer.build_author_citation_graph(min_year=2000, max_year=2024)

    # 3. Если граф пустой, попробуем другой подход
    if not analyzer.author_graph or analyzer.author_graph.number_of_nodes() == 0:
        print("\n" + "=" * 60)
        print("Граф пустой. Пробуем альтернативный подход...")
        print("=" * 60)

        # Показываем примеры данных для понимания проблемы
        if analyzer.works:
            print("\nПример работы:")
            work = analyzer.works[0]
            print(f"ID: {work.get('id')}")
            print(f"Title: {work.get('title')}")
            print(f"Authorships: {len(work.get('authorships', []))}")
            print(f"Referenced works: {len(work.get('referenced_works', []))}")

        return

    # 4. Обнаружение сообществ
    analyzer.detect_author_communities(method="connected_components")

    # 5. Обучение нулевой модели
    analyzer.fit_dcSBM()

    # 6. Поиск избыточных цитирований
    analyzer.find_excessive_citations(alpha=0.05)  # Более мягкий порог

    # 7. Обнаружение аномальных групп
    anomalous_groups = analyzer.detect_anomalous_groups(theta=0.1, min_citations=5)  # Более мягкие параметры

    # 8. Анализ и вывод результатов
    analyzer.analyze_groups(anomalous_groups, top_k=5)

    # 9. Сохранение результатов
    if anomalous_groups:
        analyzer.save_results(anomalous_groups, "author_citation_cartels.json")

    print("\n" + "=" * 60)
    print("АНАЛИЗ ЗАВЕРШЕН!")
    print("=" * 60)

    # Краткая статистика
    if anomalous_groups:
        print(f"\nИТОГОВАЯ СТАТИСТИКА:")
        print(f"  Всего групп: {len(anomalous_groups)}")
        print(f"  Максимальный размер группы: {max(g['size'] for g in anomalous_groups)}")
        print(f"  Минимальный размер группы: {min(g['size'] for g in anomalous_groups)}")
        print(f"  Средний размер группы: {np.mean([g['size'] for g in anomalous_groups]):.1f}")

        total_excessive = sum(g['excessive_citations'] for g in anomalous_groups)
        print(f"  Всего избыточных цитирований: {total_excessive}")

        # Группы с самой высокой долей избыточных цитирований
        high_ratio_groups = sorted(anomalous_groups, key=lambda x: x['excessive_ratio'], reverse=True)[:3]
        print(f"\nГруппы с самой высокой долей избыточных цитирований:")
        for idx, group in enumerate(high_ratio_groups):
            print(f"  #{idx + 1}: {group['size']} авторов, {group['excessive_ratio']:.1%} избыточных")
    else:
        print("Аномальных групп не найдено.")


if __name__ == "__main__":
    main()