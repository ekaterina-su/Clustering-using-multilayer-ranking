import json
from collections import defaultdict, Counter
import pandas as pd
from typing import Dict, List, Set, Tuple
import matplotlib.pyplot as plt


class CitationCartelDetector:
    def __init__(self, json_file_path: str):
        """
        Инициализация детектора картелей цитирования

        Args:
            json_file_path: путь к файлу openalex_data.json
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.papers = self.data['results']
        self.author_papers = defaultdict(set)  # автор -> его работы
        self.paper_authors = defaultdict(set)  # работа -> авторы
        self.citation_pairs = defaultdict(int)  # (автор1, автор2) -> кол-во цитирований
        self.author_names = {}  # id автора -> имя

        self._build_author_network()

    def _build_author_network(self):
        """Построение сети авторов на основе данных о публикациях и цитированиях"""

        # Сначала собираем всех авторов и их работы
        for paper in self.papers:
            paper_id = paper['id']

            # Получаем авторов текущей работы
            authors_in_paper = set()
            for authorship in paper.get('authorships', []):
                author_id = authorship['author']['id']
                author_name = authorship['author']['display_name']

                self.author_names[author_id] = author_name
                self.author_papers[author_id].add(paper_id)
                authors_in_paper.add(author_id)

            self.paper_authors[paper_id] = authors_in_paper

            # Для каждой пары авторов в работе отмечаем, что они соавторы
            author_list = list(authors_in_paper)
            for i in range(len(author_list)):
                for j in range(i + 1, len(author_list)):
                    # Соавторство будет учтено при анализе
                    pass

        # Анализируем цитирования
        for paper in self.papers:
            paper_id = paper['id']
            cited_works = paper.get('referenced_works', [])

            # Авторы цитирующей работы
            citing_authors = self.paper_authors.get(paper_id, set())

            # Для каждой цитируемой работы
            for cited_work in cited_works:
                # Авторы цитируемой работы
                cited_authors = self.paper_authors.get(cited_work, set())

                # Увеличиваем счетчик для каждой пары цитирующий-цитируемый
                for citing_author in citing_authors:
                    for cited_author in cited_authors:
                        # Проверяем, что это не один и тот же автор
                        if citing_author != cited_author:
                            self.citation_pairs[(citing_author, cited_author)] += 1

    def get_coauthorship_papers(self, author1: str, author2: str) -> Set[str]:
        """Получить работы, где авторы являются соавторами"""
        return self.author_papers[author1] & self.author_papers[author2]

    def is_cited_by(self, citing_author: str, cited_author: str) -> bool:
        """Проверяет, цитировал ли citing_author работы cited_author"""
        return self.citation_pairs.get((citing_author, cited_author), 0) > 0

    def detect_cartels_v1(self, threshold: int = 10) -> List[Tuple[str, str, Dict]]:
        """
        Сценарий 1: Авторы не соавторы, но много цитируют друг друга

        IsInCartelWithV1(A1, A2) = 
        IsAuthorOf(A1, P1) ∧ IsCitedThe(A1, A2) ∧ 
        IsAuthorOf(A2, P2) ∧ IsCitedThe(A2, A1) ∧ 
        ¬(IsAuthorOf(A1, P3) ∧ IsAuthorOf(A2, P3)) ∧ 
        NumberOfCites(A1, A2) ≥ 10 ∧ NumberOfCites(A2, A1) ≥ 10
        """
        cartels = []

        # Получаем все пары авторов, которые цитируют друг друга
        for (a1, a2), cites_a1_to_a2 in self.citation_pairs.items():
            # Проверяем обратное цитирование
            cites_a2_to_a1 = self.citation_pairs.get((a2, a1), 0)

            # Проверяем все условия
            if (cites_a1_to_a2 >= threshold and
                    cites_a2_to_a1 >= threshold and
                    not self.get_coauthorship_papers(a1, a2)):  # нет общих работ

                cartels.append((a1, a2, {
                    'author1_name': self.author_names.get(a1, 'Unknown'),
                    'author2_name': self.author_names.get(a2, 'Unknown'),
                    'cites_a1_to_a2': cites_a1_to_a2,
                    'cites_a2_to_a1': cites_a2_to_a1,
                    'total_cites': cites_a1_to_a2 + cites_a2_to_a1
                }))

        return cartels

    def detect_cartels_v2(self, threshold: int = 10) -> List[Tuple[str, str, Dict]]:
        """
        Сценарий 2: Авторы соавторы и много цитируют друг друга

        isInCartelWithV2(A1, A2) = 
        IsAuthorOf(A1, P1) ∧ IsCitedThe(A1, A2) ∧ 
        IsAuthorOf(A2, P2) ∧ IsCitedThe(A2, A1) ∧ 
        IsAuthorOf(A1, P3) ∧ IsAuthorOf(A2, P3) ∧ 
        NumberOfCites(A1, A2) ≥ 10 ∧ NumberOfCites(A2, A1) ≥ 10
        """
        cartels = []

        # Получаем все пары авторов, которые цитируют друг друга
        for (a1, a2), cites_a1_to_a2 in self.citation_pairs.items():
            # Проверяем обратное цитирование
            cites_a2_to_a1 = self.citation_pairs.get((a2, a1), 0)

            # Проверяем все условия
            if (cites_a1_to_a2 >= threshold and
                    cites_a2_to_a1 >= threshold and
                    self.get_coauthorship_papers(a1, a2)):  # есть общие работы

                # Получаем список совместных работ
                common_papers = list(self.get_coauthorship_papers(a1, a2))

                cartels.append((a1, a2, {
                    'author1_name': self.author_names.get(a1, 'Unknown'),
                    'author2_name': self.author_names.get(a2, 'Unknown'),
                    'cites_a1_to_a2': cites_a1_to_a2,
                    'cites_a2_to_a1': cites_a2_to_a1,
                    'total_cites': cites_a1_to_a2 + cites_a2_to_a1,
                    'common_papers': common_papers,
                    'common_papers_count': len(common_papers)
                }))

        return cartels

    def analyze_for_different_thresholds(self, thresholds: List[int]):
        """
        Анализ для разных пороговых значений
        """
        results = {
            'threshold': [],
            'v1_count': [],
            'v2_count': [],
            'v1_avg_cites': [],
            'v2_avg_cites': []
        }

        for threshold in thresholds:
            v1_cartels = self.detect_cartels_v1(threshold)
            v2_cartels = self.detect_cartels_v2(threshold)

            results['threshold'].append(threshold)
            results['v1_count'].append(len(v1_cartels))
            results['v2_count'].append(len(v2_cartels))

            # Среднее количество цитирований для найденных пар
            if v1_cartels:
                avg_v1 = sum(c[2]['total_cites'] for c in v1_cartels) / len(v1_cartels)
                results['v1_avg_cites'].append(avg_v1)
            else:
                results['v1_avg_cites'].append(0)

            if v2_cartels:
                avg_v2 = sum(c[2]['total_cites'] for c in v2_cartels) / len(v2_cartels)
                results['v2_avg_cites'].append(avg_v2)
            else:
                results['v2_avg_cites'].append(0)

        return pd.DataFrame(results)

    def print_cartel_details(self, cartels: List[Tuple], scenario_name: str, threshold: int):
        """
        Вывод детальной информации о найденных картелях
        """
        print(f"\n{'=' * 80}")
        print(f"{scenario_name} (порог = {threshold})")
        print(f"Найдено пар: {len(cartels)}")
        print('=' * 80)

        if not cartels:
            return

        # Сортируем по общему количеству цитирований
        sorted_cartels = sorted(cartels, key=lambda x: x[2]['total_cites'], reverse=True)

        for i, (a1, a2, details) in enumerate(sorted_cartels[:10], 1):  # Показываем топ-10
            print(f"\n{i}. {details['author1_name']} <-> {details['author2_name']}")
            print(f"   {details['author1_name']} -> {details['author2_name']}: {details['cites_a1_to_a2']}")
            print(f"   {details['author2_name']} -> {details['author1_name']}: {details['cites_a2_to_a1']}")
            print(f"   Всего цитирований: {details['total_cites']}")

            if 'common_papers_count' in details:
                print(f"   Совместных работ: {details['common_papers_count']}")


def main():
    # Путь к вашему JSON файлу
    json_file = 'openalex_data.json'

    # Создаем детектор
    print("Загрузка данных и построение сети авторов...")
    detector = CitationCartelDetector(json_file)

    # Анализируем для разных пороговых значений
    thresholds = [5, 10, 15, 20, 25, 30, 40, 50]

    print("\nАнализ для разных пороговых значений:")
    results_df = detector.analyze_for_different_thresholds(thresholds)
    print(results_df.to_string())

    # Детальный анализ для порога 10 (как в задании)
    print("\n" + "=" * 80)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ ДЛЯ ПОРОГА 5")
    print("=" * 80)

    # Сценарий 1
    v1_cartels = detector.detect_cartels_v1(threshold=5)
    detector.print_cartel_details(v1_cartels, "СЦЕНАРИЙ 1 (не соавторы)", 5)

    # Сценарий 2
    v2_cartels = detector.detect_cartels_v2(threshold=5)
    detector.print_cartel_details(v2_cartels, "СЦЕНАРИЙ 2 (соавторы)", 5)



if __name__ == "__main__":
    main()