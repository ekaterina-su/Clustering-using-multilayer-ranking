import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class CitationRelevanceAnalyzer:
    def __init__(self, abstracts_file: str = 'extracted_abstracts.json',
                 model_name: str = 'all-MiniLM-L6-v2'):
        """
        Инициализация анализатора релевантности цитирований

        Args:
            abstracts_file: файл с абстрактами от предыдущего шага
            model_name: название SBERT модели
        """
        print(f"Загрузка модели {model_name}...")
        self.model = SentenceTransformer(model_name)

        print(f"Загрузка данных из {abstracts_file}...")
        with open(abstracts_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.papers = self.data['papers']
        self.paper_by_id = {p['id']: p for p in self.papers}
        self.paper_by_doi = {p['doi']: p for p in self.papers if p['doi']}

        # Кэш для эмбеддингов
        self.embeddings_cache = {}

    def prepare_text_for_embedding(self, paper: Dict) -> str:
        """
        Подготовка текста статьи для эмбеддинга
        """
        texts = []

        # Добавляем название
        if paper.get('title'):
            texts.append(f"Title: {paper['title']}")

        # Добавляем абстракт
        if paper.get('abstract'):
            texts.append(f"Abstract: {paper['abstract']}")

        # Если нет ни названия, ни абстракта, используем DOI
        if not texts:
            texts.append(f"Paper: {paper.get('doi', 'Unknown')}")

        return ' '.join(texts)

    def get_embedding(self, paper_id: str) -> np.ndarray:
        """
        Получение эмбеддинга для статьи (с кэшированием)
        """
        if paper_id in self.embeddings_cache:
            return self.embeddings_cache[paper_id]

        paper = self.paper_by_id.get(paper_id)
        if not paper:
            return None

        text = self.prepare_text_for_embedding(paper)
        embedding = self.model.encode(text, convert_to_numpy=True)
        self.embeddings_cache[paper_id] = embedding

        return embedding

    def analyze_citation_relevance(self, threshold: float = 0.5) -> List[Dict]:
        """
        Анализ релевантности цитирований

        Args:
            threshold: порог семантической близости (ниже - нерелевантно)

        Returns:
            список подозрительных цитирований
        """
        suspicious_citations = []

        print("\nАнализ цитирований...")

        for paper in tqdm(self.papers, desc="Обработка статей"):
            if not paper['has_abstract']:
                continue

            citing_paper_id = paper['id']
            citing_embedding = self.get_embedding(citing_paper_id)

            if citing_embedding is None:
                continue

            cited_works = self.get_cited_works(paper)

            for cited_work_id in cited_works:
                cited_embedding = self.get_embedding(cited_work_id)

                if cited_embedding is None:
                    continue

                # Вычисляем семантическую близость
                similarity = cosine_similarity(
                    citing_embedding.reshape(1, -1),
                    cited_embedding.reshape(1, -1)
                )[0][0]

                if similarity < threshold:
                    suspicious_citations.append({
                        'citing_paper_id': citing_paper_id,
                        'citing_title': paper['title'],
                        'citing_doi': paper['doi'],
                        'cited_paper_id': cited_work_id,
                        'cited_title': self.paper_by_id.get(cited_work_id, {}).get('title', 'Unknown'),
                        'cited_doi': self.paper_by_id.get(cited_work_id, {}).get('doi', 'Unknown'),
                        'similarity_score': float(similarity),
                        'similarity_percent': float(similarity * 100)
                    })

        return suspicious_citations

    def get_cited_works(self, paper: Dict) -> List[str]:
        """
        Получение списка цитируемых работ для статьи
        Адаптируйте этот метод под структуру ваших данных
        """
        # Пока возвращаем пустой список
        return []

    def analyze_citation_patterns(self, suspicious_citations: List[Dict]):
        """
        Анализ паттернов нерелевантных цитирований
        """
        if not suspicious_citations:
            print("Нет подозрительных цитирований")
            return

        df = pd.DataFrame(suspicious_citations)

        print("\n" + "=" * 80)
        print("АНАЛИЗ ПОДОЗРИТЕЛЬНЫХ ЦИТИРОВАНИЙ")
        print("=" * 80)

        print(f"\nВсего найдено: {len(suspicious_citations)}")
        print(f"Средняя близость: {df['similarity_score'].mean():.3f}")
        print(f"Минимальная близость: {df['similarity_score'].min():.3f}")
        print(f"Максимальная близость: {df['similarity_score'].max():.3f}")

        # Статистика по цитирующим статьям
        citing_stats = df['citing_title'].value_counts().head(10)
        print("\nТоп-10 статей с подозрительными цитированиями:")
        for title, count in citing_stats.items():
            print(f"  {count}: {title[:100]}...")

        return df




class EnhancedCitationAnalyzer(CitationRelevanceAnalyzer):
    """
    Улучшенный анализатор с дополнительными метриками
    """

    def __init__(self, abstracts_file: str = 'extracted_abstracts.json',
                 openalex_file: str = 'openalex_data.json'):
        super().__init__(abstracts_file)

        # Загружаем оригинальные данные OpenAlex для получения информации о цитированиях
        with open(openalex_file, 'r', encoding='utf-8') as f:
            self.openalex_data = json.load(f)

        # Строим граф цитирований
        self.citation_graph = self.build_citation_graph()

    def build_citation_graph(self) -> Dict[str, List[str]]:
        """
        Построение графа цитирований из данных OpenAlex
        """
        citation_graph = {}

        for paper in self.openalex_data['results']:
            paper_id = paper['id']
            cited_works = paper.get('referenced_works', [])
            citation_graph[paper_id] = cited_works

        return citation_graph

    def get_cited_works(self, paper: Dict) -> List[str]:
        """
        Получение цитируемых работ из графа
        """
        paper_id = paper['id']
        return self.citation_graph.get(paper_id, [])

    def analyze_all_citations(self, thresholds: List[float] = [0.3, 0.4, 0.5, 0.6]):
        """
        Анализ для разных пороговых значений
        """
        results = {}

        for threshold in thresholds:
            print(f"\nАнализ с порогом {threshold}...")
            suspicious = self.analyze_citation_relevance(threshold=threshold)
            results[threshold] = len(suspicious)

            if suspicious:
                # Сохраняем результаты для каждого порога
                df = pd.DataFrame(suspicious)
                df = df.sort_values('similarity_score', ascending=True)

                output_file = f'suspicious_citations_th_{int(threshold * 100)}.csv'
                df.to_csv(output_file, index=False, encoding='utf-8')
                print(f"Сохранено {len(suspicious)} подозрительных цитирований в {output_file}")

        return results

    def find_extreme_cases(self, n: int = 20):
        """
        Поиск самых нерелевантных цитирований
        """
        all_suspicious = self.analyze_citation_relevance(threshold=1.0)  # Берем все

        if not all_suspicious:
            return []

        df = pd.DataFrame(all_suspicious)
        extreme_cases = df.nsmallest(n, 'similarity_score')

        print("\n" + "=" * 80)
        print(f"ТОП-{n} САМЫХ НЕРЕЛЕВАНТНЫХ ЦИТИРОВАНИЙ")
        print("=" * 80)

        for i, row in extreme_cases.iterrows():
            print(f"\n{i + 1}. Близость: {row['similarity_score']:.3f} ({row['similarity_percent']:.1f}%)")
            print(f"   Цитирующая: {row['citing_title'][:100]}...")
            print(f"   Цитируемая: {row['cited_title'][:100]}...")

        return extreme_cases

    def visualize_similarity_distribution(self, suspicious_citations: List[Dict]):
        """
        Визуализация распределения семантической близости
        """
        if not suspicious_citations:
            print("Нет данных для визуализации")
            return

        df = pd.DataFrame(suspicious_citations)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Гистограмма распределения
        axes[0, 0].hist(df['similarity_score'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Семантическая близость')
        axes[0, 0].set_ylabel('Количество цитирований')
        axes[0, 0].set_title('Распределение семантической близости')
        axes[0, 0].axvline(df['similarity_score'].mean(), color='red', linestyle='--',
                           label=f'Среднее: {df["similarity_score"].mean():.3f}')
        axes[0, 0].legend()

        # Box plot
        axes[0, 1].boxplot(df['similarity_score'])
        axes[0, 1].set_ylabel('Семантическая близость')
        axes[0, 1].set_title('Box plot семантической близости')

        # Cumulative distribution
        axes[1, 0].hist(df['similarity_score'], bins=50, cumulative=True, density=True,
                        histtype='step', linewidth=2)
        axes[1, 0].set_xlabel('Семантическая близость')
        axes[1, 0].set_ylabel('Кумулятивная вероятность')
        axes[1, 0].set_title('Кумулятивное распределение')

        # Scatter plot индексов
        axes[1, 1].scatter(range(len(df)), df.sort_values('similarity_score')['similarity_score'],
                           alpha=0.5, s=10)
        axes[1, 1].set_xlabel('Индекс цитирования')
        axes[1, 1].set_ylabel('Семантическая близость')
        axes[1, 1].set_title('Упорядоченные значения близости')

        plt.tight_layout()
        plt.savefig('citation_similarity_analysis.png', dpi=150)
        plt.show()

        print("\nГрафики сохранены в 'citation_similarity_analysis.png'")


def main():
    print("=" * 80)
    print("АНАЛИЗ РЕЛЕВАНТНОСТИ ЦИТИРОВАНИЙ С ИСПОЛЬЗОВАНИЕМ SBERT")
    print("=" * 80)

    # Инициализируем анализатор
    analyzer = EnhancedCitationAnalyzer(
        abstracts_file='extracted_abstracts.json',
        openalex_file='openalex_data.json'
    )

    # Анализ для разных порогов
    print("\n1. Анализ с разными пороговыми значениями")
    results = analyzer.analyze_all_citations(thresholds=[0.2, 0.3, 0.4, 0.5])

    # Поиск самых нерелевантных цитирований
    print("\n2. Поиск самых нерелевантных цитирований")
    extreme_cases = analyzer.find_extreme_cases(n=20)

    # Детальный анализ для порога 0.4
    print("\n3. Детальный анализ для порога 0.4")
    suspicious = analyzer.analyze_citation_relevance(threshold=0.4)

    if suspicious:
        # Анализ паттернов
        df = analyzer.analyze_citation_patterns(suspicious)

        # Визуализация
        analyzer.visualize_similarity_distribution(suspicious)

        # Сохраняем полные результаты
        output_file = 'suspicious_citations_detailed.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'threshold': 0.4,
                'total_suspicious': len(suspicious),
                'citations': suspicious
            }, f, ensure_ascii=False, indent=2)

        print(f"\nДетальные результаты сохранены в {output_file}")
    else:
        print("\nПодозрительных цитирований не найдено")

    print("\n" + "=" * 80)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 80)


if __name__ == "__main__":
    main()
