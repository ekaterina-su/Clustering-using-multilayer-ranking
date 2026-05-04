import json
import numpy as np
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class CitationCartelDetector:
    """
    Детектор картелей цитирования на основе анализа графа авторов.

    Использует комбинированный подход:
    1. Компоненты сильной связности (SCC) для выделения тесных групп
    2. Квази-иерархический анализ для оценки влияния между группами
    3. Специализированные метрики для идентификации картелей
    """

    def __init__(self, json_path, n_papers=None):
        """
        Параметры:
        - json_path: путь к файлу с данными OpenAlex
        - n_papers: количество статей для анализа (если None - все)
        """
        self.json_path = json_path
        self.load_data(json_path, n_papers)
        self.build_author_mapping()
        self.factors = {}
        self.A = None
        self.T = None
        self.clusters = None
        self.normalized_factors = {}

    def load_data(self, json_path, n_papers):
        """Загрузка и предобработка данных"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            papers_list = data
        else:
            papers_list = data.get('results', [])

        if n_papers:
            papers_list = papers_list[:n_papers]

        self.papers = papers_list
        print(f"Загружено статей: {len(self.papers)}")

    def build_author_mapping(self):
        """Создание маппинга авторов и их индексов"""
        print("\nПостроение маппинга авторов...")

        self.author_to_idx = {}
        self.author_papers = defaultdict(list)
        self.author_names = {}

        for paper_idx, paper in enumerate(self.papers):
            authorships = paper.get('authorships', [])
            for authorship in authorships:
                author = authorship.get('author', {})
                author_id = author.get('id', '')
                if author_id:
                    author_id = author_id.split('/')[-1]

                    if author_id not in self.author_to_idx:
                        self.author_to_idx[author_id] = len(self.author_to_idx)
                        author_name = author.get('display_name', 'Unknown')
                        self.author_names[author_id] = author_name

                    self.author_papers[author_id].append(paper_idx)

        self.idx_to_author = {idx: author_id for author_id, idx in self.author_to_idx.items()}
        self.n = len(self.author_to_idx)

        print(f"Найдено уникальных авторов: {self.n}")
        print(f"Примеры авторов:")
        for i, (author_id, idx) in enumerate(list(self.author_to_idx.items())[:5]):
            print(f"  {idx}: {author_id} - {self.author_names[author_id]}")

    def extract_author_ids(self, authorships):
        """Извлечение ID авторов из authorships"""
        author_ids = set()
        if not authorships:
            return author_ids

        for authorship in authorships:
            if not isinstance(authorship, dict):
                continue
            author = authorship.get('author')
            if not author or not isinstance(author, dict):
                continue
            author_id = author.get('id')
            if author_id and isinstance(author_id, str):
                try:
                    author_id = author_id.split('/')[-1]
                    if author_id:
                        author_ids.add(author_id)
                except (AttributeError, IndexError):
                    continue

        return author_ids

    def extract_reference_ids(self, referenced_works):
        """Извлечение ID цитируемых работ"""
        ref_ids = set()
        if not referenced_works:
            return ref_ids

        for ref in referenced_works:
            if ref and isinstance(ref, str):
                try:
                    ref_id = ref.split('/')[-1]
                    if ref_id:
                        ref_ids.add(ref_id)
                except (AttributeError, IndexError):
                    continue

        return ref_ids

    def compute_factor_matrices(self):
        """Вычисление факторов влияния между авторами"""
        print("\nВычисление факторов влияния между авторами...")
        print(f"Всего авторов для анализа: {self.n}")

        n_authors = self.n
        if n_authors == 0:
            print("Нет данных для обработки!")
            return

        # Инициализируем матрицы факторов
        self.factors = {
            'common_refs': np.zeros((n_authors, n_authors)),
            'common_authors': np.zeros((n_authors, n_authors)),
            'direct_citation': np.zeros((n_authors, n_authors)),
            'citation_count': np.zeros((n_authors, n_authors))
        }

        # Предвычисляем данные по статьям
        print("  Предобработка данных статей...")
        paper_refs = []
        paper_authors = []
        paper_cited_by = []

        for paper in self.papers:
            refs = self.extract_reference_ids(paper.get('referenced_works', []))
            paper_refs.append(refs)

            authors = self.extract_author_ids(paper.get('authorships', []))
            paper_authors.append(authors)

            cited_by = paper.get('cited_by_count', 0)
            if cited_by is None:
                cited_by = 0
            paper_cited_by.append(cited_by)

        # Собираем все цитирования между статьями
        print("  Анализ цитирований между статьями...")
        paper_citations = defaultdict(int)

        for i, paper in enumerate(self.papers):
            cited_papers = paper.get('referenced_works', [])
            for cited in cited_papers:
                if cited and isinstance(cited, str):
                    cited_id = cited.split('/')[-1]
                    for j, p in enumerate(self.papers):
                        if p['id'].split('/')[-1] == cited_id:
                            paper_citations[(i, j)] = 1
                            break

        # Строим связи между авторами
        print("  Построение связей между авторами...")
        author_indices = list(range(n_authors))

        for a1_idx in range(n_authors):
            if a1_idx % 100 == 0:
                print(f"    Обработано {a1_idx}/{n_authors} авторов")

            author1_id = self.idx_to_author[a1_idx]
            papers1 = self.author_papers[author1_id]

            for a2_idx in range(a1_idx + 1, n_authors):
                author2_id = self.idx_to_author[a2_idx]
                papers2 = self.author_papers[author2_id]

                # Фактор 1: общие ссылки
                all_refs1 = set()
                all_refs2 = set()
                for p1 in papers1:
                    all_refs1.update(paper_refs[p1])
                for p2 in papers2:
                    all_refs2.update(paper_refs[p2])

                common_refs = len(all_refs1.intersection(all_refs2))
                if common_refs > 0:
                    self.factors['common_refs'][a1_idx, a2_idx] = common_refs
                    self.factors['common_refs'][a2_idx, a1_idx] = common_refs

                # Фактор 2: соавторство
                coauthored = False
                for p1 in papers1:
                    for p2 in papers2:
                        if p1 == p2:
                            coauthored = True
                            break
                    if coauthored:
                        break

                if coauthored:
                    self.factors['common_authors'][a1_idx, a2_idx] = 1
                    self.factors['common_authors'][a2_idx, a1_idx] = 1

                # Фактор 3: прямое цитирование
                citations_12 = 0
                citations_21 = 0

                for p1 in papers1:
                    for p2 in papers2:
                        if (p1, p2) in paper_citations:
                            citations_12 += 1
                        if (p2, p1) in paper_citations:
                            citations_21 += 1

                if citations_12 > 0:
                    self.factors['direct_citation'][a1_idx, a2_idx] = citations_12
                if citations_21 > 0:
                    self.factors['direct_citation'][a2_idx, a1_idx] = citations_21

                # Фактор 4: влияние через цитируемость
                total_citations_a2 = sum(paper_cited_by[p] for p in papers2)
                if total_citations_a2 > 0:
                    self.factors['citation_count'][a1_idx, a2_idx] = total_citations_a2

                total_citations_a1 = sum(paper_cited_by[p] for p in papers1)
                if total_citations_a1 > 0:
                    self.factors['citation_count'][a2_idx, a1_idx] = total_citations_a1

        print("  Факторы между авторами вычислены!")

        # Статистика
        print("\nСтатистика по факторам (связи между авторами):")
        for factor_name, matrix in self.factors.items():
            non_zero = matrix[matrix > 0]
            print(f"  {factor_name}: {len(non_zero)} ненулевых, "
                  f"макс: {non_zero.max() if len(non_zero) > 0 else 0}")

    def normalize_factors(self, method='robust'):
        """Нормализация факторов к [0, 1]"""
        print("\nНормализация факторов...")

        self.normalized_factors = {}

        for factor_name, matrix in self.factors.items():
            print(f"  Нормализация {factor_name}...")

            if method == 'robust':
                non_zero = matrix[matrix > 0]
                if len(non_zero) > 0:
                    q99 = np.percentile(non_zero, 99)
                    matrix_clipped = np.clip(matrix, 0, q99)
                    if q99 > 0:
                        normalized = matrix_clipped / q99
                    else:
                        normalized = matrix_clipped
                else:
                    normalized = matrix
            elif method == 'minmax':
                max_val = matrix.max()
                normalized = matrix / max_val if max_val > 0 else matrix
            else:
                normalized = matrix

            self.normalized_factors[factor_name] = normalized

    def build_influence_matrix(self, alphas=None, threshold_percentile=95):
        """Построение матрицы влияния между авторами"""
        print("\nПостроение матрицы влияния между авторами...")

        if alphas is None:
            alphas = {
                'common_refs': 0.20,
                'common_authors': 0.15,
                'direct_citation': 0.40,
                'citation_count': 0.25
            }

        self.alphas = alphas

        if not self.normalized_factors:
            print("  Сначала выполняем нормализацию...")
            self.normalize_factors()

        self.A = np.zeros((self.n, self.n))

        for factor_name, alpha in alphas.items():
            if factor_name in self.normalized_factors:
                self.A += alpha * self.normalized_factors[factor_name]
                print(f"  Добавлен фактор {factor_name} с весом {alpha}")

        # Отсечение слабых связей
        if threshold_percentile < 100:
            non_zero_values = self.A[self.A > 0]
            if len(non_zero_values) > 0:
                threshold = np.percentile(non_zero_values, threshold_percentile)
                self.A[self.A < threshold] = 0

                print(f"\n  Отсечение по {threshold_percentile}-му процентилю:")
                print(f"    Порог: {threshold:.6f}")
                print(f"    Осталось связей: {np.sum(self.A > 0)}")

        return self.A

    def transitive_closure(self):
        """Вычисление транзитивного замыкания графа авторов"""
        print("\nВычисление транзитивного замыкания для графа авторов...")

        if self.A is None:
            print("  Ошибка: матрица влияния не построена!")
            return None

        T = (self.A > 0).astype(int)
        np.fill_diagonal(T, 1)

        n = self.n
        for k in range(n):
            if k % 100 == 0:
                print(f"  Итерация {k}/{n}")

            row_k = T[k, :]
            col_k = T[:, k]

            for i in range(n):
                if col_k[i]:
                    T[i, :] = np.logical_or(T[i, :], row_k)

        self.T = T
        print(f"  Транзитивное замыкание завершено")
        print(f"  Плотность матрицы достижимости: {np.sum(T) / (n * n):.4f}")

        return T

    def find_clusters(self):
        """
        Поиск кластеров авторов на основе компонент сильной связности (SCC)

        Это основной метод кластеризации для поиска картелей.
        SCC идеально подходит для выявления групп с взаимным цитированием.
        """
        print("\n" + "=" * 70)
        print("ПОИСК КЛАСТЕРОВ АВТОРОВ (SCC)")
        print("=" * 70)

        if not hasattr(self, 'T') or self.T is None:
            print("  Вычисляем транзитивное замыкание...")
            self.transitive_closure()

        n = self.n
        visited = np.zeros(n, dtype=bool)
        clusters = []

        print("  Поиск компонент сильной связности...")
        for i in range(n):
            if not visited[i]:
                cluster = []
                for j in range(n):
                    if not visited[j] and self.T[i, j] == 1 and self.T[j, i] == 1:
                        cluster.append(j)
                        visited[j] = True

                if cluster:
                    clusters.append(cluster)

        # Сортируем по размеру
        clusters.sort(key=len, reverse=True)
        self.clusters = clusters

        print(f"\nРезультаты кластеризации (SCC):")
        print(f"  Всего кластеров: {len(clusters)}")
        print(f"  Размеры топ-10 кластеров: {[len(c) for c in clusters[:10]]}")

        # Статистика по размерам
        sizes = [len(c) for c in clusters]
        print(f"  Медианный размер: {np.median(sizes):.1f}")
        print(f"  Средний размер: {np.mean(sizes):.1f}")
        print(f"  Кластеров размера 1: {sum(1 for s in sizes if s == 1)}")
        print(f"  Кластеров размера 2-5: {sum(1 for s in sizes if 2 <= s <= 5)}")
        print(f"  Кластеров размера 6-15: {sum(1 for s in sizes if 6 <= s <= 15)}")
        print(f"  Кластеров размера >15: {sum(1 for s in sizes if s > 15)}")

        return clusters

    def analyze_cluster_hierarchy(self):
        """
        Квази-иерархический анализ кластеров

        Вычисляет уровень влияния каждого кластера в сети.
        Кластеры с высоким влиянием - потенциальные "лидеры мнений".
        Кластеры с низким влиянием но высокой внутренней связностью - потенциальные картели.
        """
        print("\n" + "=" * 70)
        print("КВАЗИ-ИЕРАРХИЧЕСКИЙ АНАЛИЗ КЛАСТЕРОВ")
        print("=" * 70)

        if not self.clusters:
            print("Сначала выполните find_clusters()!")
            return

        # Вычисляем метрики влияния для каждого кластера
        cluster_metrics = []

        for cluster_idx, cluster in enumerate(self.clusters):
            if len(cluster) < 2:
                continue

            metrics = self._compute_cluster_metrics(cluster)
            metrics['cluster_idx'] = cluster_idx
            cluster_metrics.append(metrics)

        # Сортируем по влиятельности
        cluster_metrics.sort(key=lambda x: x['influence_score'], reverse=True)

        # Выводим иерархию
        print(f"\n{'=' * 70}")
        print("ИЕРАРХИЯ КЛАСТЕРОВ ПО ВЛИЯНИЮ")
        print(f"{'=' * 70}")
        print(f"{'Уровень':<8} {'Кластер':<10} {'Размер':<8} {'Влияние':<10} {'Взаимность':<12} {'Изоляция':<10}")
        print("-" * 58)

        for level, metrics in enumerate(cluster_metrics[:20]):
            print(f"{level:<8} {metrics['cluster_idx']:<10} {metrics['size']:<8} "
                  f"{metrics['influence_score']:<10.1f} {metrics['reciprocity']:<12.3f} "
                  f"{metrics['isolation']:<10.3f}")

        self.cluster_metrics = cluster_metrics

        # Определяем потенциальные картели
        # Картель = высокая взаимность + низкое внешнее влияние + небольшой размер
        print(f"\n{'=' * 70}")
        print("ПОТЕНЦИАЛЬНЫЕ КАРТЕЛИ (критерии отбора)")
        print(f"{'=' * 70}")

        cartel_candidates = []
        for metrics in cluster_metrics:
            if (3 <= metrics['size'] <= 15 and  # небольшой размер
                    metrics['reciprocity'] > 0.5 and  # высокая взаимность
                    metrics['external_influence'] < 2.0):  # низкое внешнее влияние

                cartel_candidates.append(metrics)
                print(f"Кластер {metrics['cluster_idx']}: размер={metrics['size']}, "
                      f"взаимность={metrics['reciprocity']:.3f}, "
                      f"внешнее влияние={metrics['external_influence']:.1f}")

        if not cartel_candidates:
            print("  Потенциальные картели не найдены по строгим критериям")

        return cluster_metrics

    def _compute_cluster_metrics(self, cluster):
        """Вычисление метрик для кластера"""
        n = len(cluster)

        # Влиятельность: сколько других узлов достижимо из этого кластера
        influence_scores = []
        for idx in cluster:
            reachable = np.sum(self.T[idx, :])
            influence_scores.append(reachable)
        avg_influence = np.mean(influence_scores)

        # Внутренняя связность
        internal_edges = 0
        possible_internal = n * (n - 1)
        for i in cluster:
            for j in cluster:
                if i != j and self.A[i, j] > 0:
                    internal_edges += 1
        density = internal_edges / possible_internal if possible_internal > 0 else 0

        # Взаимность
        reciprocal_pairs = 0
        possible_pairs = n * (n - 1) / 2
        for i_idx, i in enumerate(cluster):
            for j in cluster[i_idx + 1:]:
                if self.A[i, j] > 0 and self.A[j, i] > 0:
                    reciprocal_pairs += 1
        reciprocity = reciprocal_pairs / possible_pairs if possible_pairs > 0 else 0

        # Внешнее влияние (сколько внешних узлов цитируют этот кластер)
        external_in = 0
        for i in cluster:
            for j in range(self.n):
                if j not in cluster and self.A[j, i] > 0:
                    external_in += 1
        external_in_per_author = external_in / n

        # Изоляция (отношение внешних связей к внутренним)
        total_external = 0
        for i in cluster:
            for j in range(self.n):
                if j not in cluster:
                    if self.A[i, j] > 0:
                        total_external += 1
                    if self.A[j, i] > 0:
                        total_external += 1
        isolation = total_external / (internal_edges + 1)

        return {
            'size': n,
            'influence_score': avg_influence,
            'density': density,
            'reciprocity': reciprocity,
            'external_influence': external_in_per_author,
            'isolation': isolation,
            'internal_edges': internal_edges
        }

    def save_clusters_to_file(self, filename='clusters_hierarchical.txt'):
        """Сохранение результатов кластеризации в файл"""
        print(f"\nСохранение результатов в {filename}...")

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ АВТОРОВ\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Всего авторов: {self.n}\n")
            f.write(f"Всего кластеров: {len(self.clusters)}\n\n")

            f.write("РАЗМЕРЫ КЛАСТЕРОВ:\n")
            f.write("-" * 30 + "\n")

            for i, cluster in enumerate(self.clusters):
                f.write(f"Кластер {i}: {len(cluster)} авторов\n")

            f.write("\n\nСОСТАВЫ КЛАСТЕРОВ:\n")
            f.write("=" * 70 + "\n\n")

            for i, cluster in enumerate(self.clusters):
                f.write(f"Кластер {i} (размер: {len(cluster)}):\n")
                f.write("-" * 40 + "\n")

                cluster_authors = sorted([
                    self.author_names[self.idx_to_author[idx]]
                    for idx in cluster
                ])

                for j, author in enumerate(cluster_authors, 1):
                    f.write(f"  {j:3d}. {author}\n")
                f.write("\n")

        print(f"Результаты сохранены в {filename}")

    def save_author_cluster_mapping(self, filename='author_cluster_mapping.txt'):
        """Сохранение маппинга автор -> кластер"""
        print(f"\nСохранение маппинга авторов в {filename}...")

        author_to_cluster = {}
        for cluster_idx, cluster in enumerate(self.clusters):
            for author_idx in cluster:
                author_id = self.idx_to_author[author_idx]
                author_to_cluster[author_id] = cluster_idx

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("МАППИНГ АВТОР -> КЛАСТЕР\n")
            f.write("=" * 50 + "\n\n")

            for author_id in sorted(self.author_to_idx.keys()):
                author_name = self.author_names[author_id]
                cluster_idx = author_to_cluster.get(author_id, -1)
                f.write(f"{author_name} -> Кластер {cluster_idx}\n")

        print(f"Маппинг сохранен в {filename}")


class CartelFinder:
    """Класс для детального анализа потенциальных картелей"""

    def __init__(self, detector, percentile=99):
        self.detector = detector
        self.percentile = percentile
        self.A = detector.A
        self.n = detector.n
        self.author_names = detector.author_names
        self.idx_to_author = detector.idx_to_author

    def analyze_small_clusters(self, min_size=3, max_size=15):
        """
        Анализ маленьких кластеров (потенциальных картелей)
        """
        print("\n" + "=" * 80)
        print(f"АНАЛИЗ МАЛЕНЬКИХ КЛАСТЕРОВ (размер {min_size}-{max_size})")
        print("=" * 80)

        small_clusters = []
        for cluster in self.detector.clusters:
            if min_size <= len(cluster) <= max_size:
                small_clusters.append(cluster)

        small_clusters.sort(key=len, reverse=True)
        print(f"Всего маленьких кластеров: {len(small_clusters)}")

        if len(small_clusters) == 0:
            print("\n❌ Маленькие кластеры не найдены!")
            return []

        candidates = []
        for cluster_idx_in_small, cluster in enumerate(small_clusters):
            original_cluster_idx = self.detector.clusters.index(cluster)

            print(f"\n{'=' * 60}")
            print(f"КЛАСТЕР {original_cluster_idx} (размер: {len(cluster)})")
            print(f"{'=' * 60}")

            # Анализ авторов
            print("\n📚 АВТОРЫ В КЛАСТЕРЕ:")
            for author_idx in cluster[:10]:
                author_id = self.idx_to_author[author_idx]
                author_name = self.author_names.get(author_id, "Unknown")
                n_papers = len(self.detector.author_papers.get(author_id, []))
                print(f"  • {author_name} — {n_papers} статей")

            if len(cluster) > 10:
                print(f"  ... и еще {len(cluster) - 10} авторов")

            # Анализ связей
            n = len(cluster)
            possible_edges = n * (n - 1)
            actual_edges = 0
            reciprocal_count = 0
            reciprocal_pairs = []

            for i in cluster:
                for j in cluster:
                    if i != j and self.A[i, j] > 0:
                        actual_edges += 1
                    if i < j and self.A[i, j] > 0 and self.A[j, i] > 0:
                        reciprocal_count += 1
                        reciprocal_pairs.append((i, j))

            density = actual_edges / possible_edges if possible_edges > 0 else 0
            reciprocity = reciprocal_count / (n * (n - 1) / 2) if n > 1 else 0

            print(f"\n🔗 АНАЛИЗ СВЯЗЕЙ:")
            print(f"  Плотность графа: {density:.3f} ({actual_edges}/{possible_edges} ребер)")
            print(f"  Взаимных пар: {reciprocal_count}")
            print(f"  Доля взаимных пар: {reciprocity:.3f}")

            strengths = []
            for i in cluster:
                for j in cluster:
                    if i != j and self.A[i, j] > 0:
                        strengths.append(self.A[i, j])

            if strengths:
                print(f"  Средняя сила связи: {np.mean(strengths):.3f}")

            # Внешние связи
            external_in = 0
            external_out = 0
            for i in cluster:
                for j in range(self.n):
                    if j not in cluster:
                        if self.A[j, i] > 0:
                            external_in += 1
                        if self.A[i, j] > 0:
                            external_out += 1

            ext_in_per_author = external_in / len(cluster)
            ext_out_per_author = external_out / len(cluster)
            isolation = (external_in + external_out) / (actual_edges + 1)

            print(f"\n🌐 ВНЕШНИЕ СВЯЗИ:")
            print(f"  Внешних цитирований НА кластер: {ext_in_per_author:.1f} на автора")
            print(f"  Внешних цитирований ИЗ кластера: {ext_out_per_author:.1f} на автора")
            print(f"  Коэффициент изоляции: {isolation:.3f}")

            # Критерии картеля
            is_cartel = True
            reasons = []

            if density >= 0.3:
                reasons.append(f"✅ плотность {density:.3f} ≥ 0.3")
            else:
                is_cartel = False
                reasons.append(f"❌ плотность {density:.3f} < 0.3")

            if reciprocity >= 0.2:
                reasons.append(f"✅ взаимность {reciprocity:.3f} ≥ 0.2")
            else:
                is_cartel = False
                reasons.append(f"❌ взаимность {reciprocity:.3f} < 0.2")

            if ext_in_per_author <= 5:
                reasons.append(f"✅ внешних цитирований {ext_in_per_author:.1f} ≤ 5")
            else:
                is_cartel = False
                reasons.append(f"❌ слишком много внешних цитирований ({ext_in_per_author:.1f} > 5)")

            print(f"\n🎯 КРИТЕРИИ КАРТЕЛЯ:")
            print("  " + "\n  ".join(reasons))

            if is_cartel:
                print(f"\n  🔴 ЭТОТ КЛАСТЕР ЯВЛЯЕТСЯ ПОТЕНЦИАЛЬНЫМ КАРТЕЛЕМ!")
                candidates.append({
                    'cluster_idx': original_cluster_idx,
                    'size': len(cluster),
                    'density': density,
                    'reciprocity': reciprocity,
                    'external_in': ext_in_per_author,
                    'external_out': ext_out_per_author,
                    'isolation': isolation
                })
            else:
                print(f"\n  🟢 НЕ является картелем")

        # Итоги
        print(f"\n{'=' * 80}")
        print("ИТОГИ АНАЛИЗА")
        print(f"{'=' * 80}")
        print(f"Всего проанализировано маленьких кластеров: {len(small_clusters)}")
        print(f"Найдено потенциальных картелей: {len(candidates)}")

        if candidates:
            print("\nПотенциальные картели:")
            for c in candidates:
                print(f"  • Кластер {c['cluster_idx']} (размер {c['size']}): "
                      f"плотность={c['density']:.2f}, "
                      f"взаимность={c['reciprocity']:.2f}")

        return candidates

    def save_cartels_to_file(self, candidates, filename='potential_cartels.json', verbose=False):
        """Сохранение информации о потенциальных картелях"""
        print(f"\n💾 Сохранение результатов в {filename}...")

        results = {
            'total_candidates': len(candidates),
            'threshold_percentile': self.percentile,
            'candidates': []
        }

        for candidate in candidates:
            cluster_idx = candidate['cluster_idx']
            cluster = self.detector.clusters[cluster_idx]

            authors_info = []
            for author_idx in cluster:
                author_id = self.idx_to_author[author_idx]
                author_name = self.author_names.get(author_id, "Unknown")
                n_papers = len(self.detector.author_papers.get(author_id, []))

                authors_info.append({
                    'author_id': author_id,
                    'name': author_name,
                    'index': int(author_idx),
                    'n_papers': n_papers
                })

            candidate_info = {
                'cluster_index': int(cluster_idx),
                'size': candidate['size'],
                'density': float(candidate['density']),
                'reciprocity': float(candidate['reciprocity']),
                'external_in_per_author': float(candidate['external_in']),
                'external_out_per_author': float(candidate['external_out']),
                'isolation': float(candidate['isolation']),
                'authors': authors_info
            }

            results['candidates'].append(candidate_info)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Текстовая версия
        txt_filename = filename.replace('.json', '.txt')
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("ПОТЕНЦИАЛЬНЫЕ КАРТЕЛИ ЦИТИРОВАНИЯ\n")
            f.write("=" * 80 + "\n\n")

            for i, candidate in enumerate(results['candidates']):
                f.write(f"КАРТЕЛЬ {i + 1} (кластер {candidate['cluster_index']})\n")
                f.write(f"{'-' * 40}\n")
                f.write(f"Размер: {candidate['size']} авторов\n")
                f.write(f"Плотность связей: {candidate['density']:.3f}\n")
                f.write(f"Взаимность: {candidate['reciprocity']:.3f}\n")
                f.write(f"Коэффициент изоляции: {candidate['isolation']:.3f}\n")
                f.write(f"Внешних цитирований на автора: {candidate['external_in_per_author']:.1f}\n\n")

                f.write("АВТОРЫ В КАРТЕЛЕ:\n")
                for author in candidate['authors']:
                    f.write(f"  • {author['name']} ({author['author_id']}) — {author['n_papers']} статей\n")
                f.write("\n" + "=" * 80 + "\n\n")

        print(f"✅ Результаты сохранены в {filename} и {txt_filename}")
        return results

def load_clusters_and_analyze():
    """Загружает сохраненные кластеры и анализирует авторов CIDRE"""

    cidre_author_ids = ['A5100405033', 'A5049853160']
    found_authors = []  # Инициализируем переменную здесь!

    print("=" * 80)
    print("АНАЛИЗ АВТОРОВ CIDRE В СОХРАНЕННЫХ КЛАСТЕРАХ")
    print("=" * 80)

    # Загружаем данные об авторах
    try:
        with open('clusters_hierarchical.txt', 'r', encoding='utf-8') as f:
            content = f.read()

        with open('potential_cartels.json', 'r', encoding='utf-8') as f:
            cartels_data = json.load(f)

        with open('author_cluster_mapping.txt', 'r', encoding='utf-8') as f:
            mapping_lines = f.readlines()
    except FileNotFoundError as e:
        print(f"❌ Файл не найден: {e}")
        print("Сначала запустите основной анализ!")
        return

    # Парсим маппинг автор -> кластер
    author_to_cluster = {}
    for line in mapping_lines:
        if '->' in line:
            parts = line.strip().split('->')
            if len(parts) == 2:
                author_name = parts[0].strip()
                cluster_str = parts[1].strip()
                try:
                    cluster_id = int(cluster_str.split()[-1])
                    author_to_cluster[author_name] = cluster_id
                except:
                    pass

    print(f"\nЗагружен маппинг для {len(author_to_cluster)} авторов")

    # Загружаем оригинальные данные чтобы найти имена по ID
    try:
        with open('openalex_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ Файл openalex_data.json не найден!")
        return

    if isinstance(data, dict) and 'results' in data:
        papers = data['results']
    else:
        papers = data

    # Строим маппинг ID -> имя
    id_to_name = {}
    for paper in papers:
        for authorship in paper.get('authorships', []):
            author = authorship.get('author', {})
            author_id = author.get('id', '')
            if author_id:
                clean_id = author_id.split('/')[-1]
                author_name = author.get('display_name', 'Unknown')
                id_to_name[clean_id] = author_name

    # Анализируем каждого автора CIDRE
    print("\n" + "=" * 60)
    print("ПОИСК АВТОРОВ CIDRE")
    print("=" * 60)

    for author_id in cidre_author_ids:
        clean_id = author_id.split('/')[-1] if '/' in author_id else author_id

        if clean_id in id_to_name:
            author_name = id_to_name[clean_id]
            found_authors.append(clean_id)  # Добавляем в список найденных
            print(f"✅ {author_name} ({clean_id}) - найден в данных")
        else:
            print(f"❌ {clean_id} - НЕ найден в данных")

    # Теперь анализируем кластер 0
    print("\n" + "=" * 80)
    print("АНАЛИЗ КЛАСТЕРА 0")
    print("=" * 80)

    # Подсчитываем количество авторов в каждом кластере из сохраненного файла
    cluster_sizes = {}
    current_cluster = None
    current_size = 0

    for line in content.split('\n'):
        if line.startswith('Кластер ') and '(размер:' in line:
            # Сохраняем предыдущий кластер
            if current_cluster is not None:
                cluster_sizes[current_cluster] = current_size

            # Начинаем новый кластер
            parts = line.split('(размер:')
            if len(parts) == 2:
                cluster_name = parts[0].strip()
                size_str = parts[1].replace('):', '').strip()
                try:
                    current_size = int(size_str)
                    # Извлекаем номер кластера
                    cluster_num = int(cluster_name.split()[1])
                    current_cluster = cluster_num
                except:
                    current_cluster = None
                    current_size = 0

    # Сохраняем последний кластер
    if current_cluster is not None:
        cluster_sizes[current_cluster] = current_size

    # Выводим информацию о кластере 0
    if 0 in cluster_sizes:
        print(f"\n📊 КЛАСТЕР 0:")
        print(f"  Размер: {cluster_sizes[0]} авторов")
        print(f"  Всего кластеров: {len(cluster_sizes)}")
        print(f"  Средний размер кластера: {sum(cluster_sizes.values()) / len(cluster_sizes):.1f}")

        # Проверяем авторов CIDRE в кластере 0
        cidre_in_cluster_0 = []
        for author_id in found_authors:
            author_name = id_to_name.get(author_id, "Unknown")
            if author_name in author_to_cluster and author_to_cluster[author_name] == 0:
                cidre_in_cluster_0.append((author_id, author_name))

        print(f"\n⭐ АВТОРЫ CIDRE В КЛАСТЕРЕ 0:")
        if cidre_in_cluster_0:
            for author_id, author_name in cidre_in_cluster_0:
                print(f"  ✅ {author_name} ({author_id}) - находится в кластере 0")
                print(f"     Это самый влиятельный кластер в сети!")
                print(f"     Авторы этого кластера цитируют максимальное число других авторов")
        else:
            print(f"  ❌ Авторы CIDRE не найдены в кластере 0")

            # Ищем, в каких кластерах они находятся
            for author_id in found_authors:
                author_name = id_to_name.get(author_id, "Unknown")
                if author_name in author_to_cluster:
                    cluster = author_to_cluster[author_name]
                    print(f"  {author_name} находится в кластере {cluster}")

        # Анализ потенциальных картелей
        if cartels_data and 'candidates' in cartels_data:
            print(f"\n🔍 ПРОВЕРКА НА КАРТЕЛИ:")

            # Проверяем, является ли кластер 0 картелем
            cluster_0_is_cartel = False
            for candidate in cartels_data['candidates']:
                if candidate['cluster_index'] == 0:
                    cluster_0_is_cartel = True
                    print(f"  ⚠️ Кластер 0 идентифицирован как ПОТЕНЦИАЛЬНЫЙ КАРТЕЛЬ!")
                    print(f"     Плотность: {candidate['density']:.3f}")
                    print(f"     Взаимность: {candidate['reciprocity']:.3f}")
                    print(f"     Изоляция: {candidate.get('isolation', 'N/A')}")
                    break

            if not cluster_0_is_cartel:
                print(f"  ✅ Кластер 0 НЕ является картелем (слишком большой или недостаточно изолирован)")
                print(f"     Размер кластера 0 ({cluster_sizes[0]}) превышает типичный размер картеля (3-15)")

            # Ищем авторов CIDRE в других картелях
            for candidate in cartels_data['candidates']:
                if candidate['cluster_index'] != 0:
                    # Проверяем, есть ли авторы CIDRE в этом картеле
                    for author in candidate.get('authors', []):
                        if author['author_id'] in found_authors:
                            print(
                                f"\n  ⚠️ {author['name']} также найден в картеле {candidate['cluster_index']}!")
                            print(f"     Размер картеля: {candidate['size']}")
                            print(f"     Плотность: {candidate['density']:.3f}")
                            print(f"     Взаимность: {candidate['reciprocity']:.3f}")

    else:
        print("❌ Кластер 0 не найден в данных!")

    # Выводим статистику по всем кластерам
    print(f"\n{'=' * 60}")
    print("СТАТИСТИКА ПО ВСЕМ КЛАСТЕРАМ:")
    print(f"{'=' * 60}")

    sorted_sizes = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    print(f"Топ-10 крупнейших кластеров:")
    for cluster_num, size in sorted_sizes[:10]:
        percentage = size / sum(cluster_sizes.values()) * 100
        print(f"  Кластер {cluster_num}: {size} авторов ({percentage:.1f}%)")

    # Ищем всех авторов CIDRE во всех кластерах
    print(f"\n{'=' * 60}")
    print("ПОЛНЫЙ АНАЛИЗ АВТОРОВ CIDRE:")
    print(f"{'=' * 60}")

    for author_id in found_authors:
        author_name = id_to_name.get(author_id, "Unknown")
        print(f"\nАвтор: {author_name} ({author_id})")

        if author_name in author_to_cluster:
            cluster = author_to_cluster[author_name]
            print(f"  Кластер: {cluster}")
            if cluster in cluster_sizes:
                print(f"  Размер кластера: {cluster_sizes[cluster]} авторов")
                print(f"  Доля от сети: {cluster_sizes[cluster] / sum(cluster_sizes.values()) * 100:.1f}%")

            # Характеристика кластера
            if cluster == 0:
                print(f"  📌 Характеристика: САМЫЙ ВЛИЯТЕЛЬНЫЙ КЛАСТЕР")
                print(f"     Авторы этого кластера цитируют наибольшее число других авторов")
                print(f"     Это может указывать на то, что они являются лидерами мнений")
                print(f"     или активно участвуют в научной коммуникации")
            elif cluster_sizes.get(cluster, 0) <= 15:
                print(f"  📌 Характеристика: МАЛЕНЬКИЙ КЛАСТЕР")
                print(f"     Потенциально может быть картелем при высокой взаимности цитирований")
            else:
                print(f"  📌 Характеристика: СРЕДНИЙ/КРУПНЫЙ КЛАСТЕР")
                print(f"     Типичное научное сообщество")
        else:
            print(f"  ❌ Не найден ни в одном кластере")


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

if __name__ == "__main__":
    # Инициализация
    print("ЗАПУСК АНАЛИЗА КАРТЕЛЕЙ ЦИТИРОВАНИЯ")
    print("=" * 80)

    detector = CitationCartelDetector('openalex_data.json', n_papers=2953)

    # Вычисление факторов между авторами
    detector.compute_factor_matrices()

    # Нормализация
    detector.normalize_factors(method='robust')

    # Построение матрицы влияния
    optimal_percentile = 99
    detector.build_influence_matrix(threshold_percentile=optimal_percentile)

    # Транзитивное замыкание
    detector.transitive_closure()

    # Кластеризация (SCC)
    clusters = detector.find_clusters()

    # Квази-иерархический анализ
    cluster_metrics = detector.analyze_cluster_hierarchy()

    # Сохранение результатов кластеризации
    detector.save_clusters_to_file('clusters_hierarchical.txt')
    detector.save_author_cluster_mapping('author_cluster_mapping.txt')

    # Поиск картелей
    finder = CartelFinder(detector, percentile=optimal_percentile)
    candidates = finder.analyze_small_clusters(min_size=3, max_size=15)

    if candidates:
        finder.save_cartels_to_file(candidates, 'potential_cartels.json', verbose=True)
    else:
        print("\n❌ Потенциальные картели не найдены")

    print("\n" + "=" * 80)

    load_clusters_and_analyze()