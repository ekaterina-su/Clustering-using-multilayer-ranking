# Кластеризация на основе многослойного ранжирования
Программная реализация методов выявления картелей цитирования в научных сетях. 
Разработано в рамках магистерской диссертации на тему: 
"Кластеризация на основе многослойного ранжирования".

## Аннотация

Данный репозиторий содержит реализацию четырх методов анализа для 
двуслойной сети «авторы – публикации»:

1. **Пороговый алгоритм** — поиск пар авторов с аномально высокой интенсивностью 
   взаимных цитирований (с разделением на соавторов и не-соавторов)
2. **Адаптированный CIDRE** — статистический подход на основе 
   degree-corrected Stochastic Block Model для выявления избыточных цитирований
3. **Иерархическая кластеризация** — выделение компонент сильной связности 
   и анализ структуры влияния
4. **Семантическая валидация** — проверка содержательной обоснованности цитирований 
   с помощью SBERT-эмбеддингов

## Архитектура
OpenAlex JSON → Abstract Extractor → SBERT Embeddings → Semantic Validation

↓

Author Citation Graph → CIDRE → Anomalous Groups

↓

Influence Matrix → Transitive Closure → Hierarchical Clustering

↓

Threshold Rules → Suspected Pairs

## Установка

```bash
git clone https://github.com/ekaterina-su/Сlustering-using-multilayer-ranking.git
cd Сlustering-using-multilayer-ranking
python -m venv venv
source venv/bin/activate  # или venv\Scripts\activate на Windows
pip install -r requirements.txt
```

## Данные

Для тестирования в репозитории находится файл `openalex_data.json` (500 статей).

**Полные данные** (исходная выборка на 5000 статей) можно получить по запросу у автора.

Структура входного JSON-файла:
```json
{
  "results": [
    {
      "id": "https://openalex.org/W123456789",
      "title": "Название статьи",
      "publication_year": 2020,
      "authorships": [
        {
          "author": {
            "id": "https://openalex.org/A1234567",
            "display_name": "Имя Автора"
          }
        }
      ],
      "referenced_works": [
        "https://openalex.org/W987654321"
      ],
      "abstract_inverted_index": {
        "word": [0, 5, 10],
        "another": [1, 3]
      }
    }
  ]
}
```
## Запуск методов
1. **Пороговый алгоритм**
```bash
python -c "
from src.detection.threshold import CitationCartelDetector
detector = CitationCartelDetector('data_sample.json')
v1 = detector.detect_cartels_v1(threshold=5)
v2 = detector.detect_cartels_v2(threshold=5)
print(f'Сценарий 1 (не соавторы): {len(v1)} пар')
print(f'Сценарий 2 (соавторы): {len(v2)} пар')
```
2. **Адаптированный CIDRE**
```bash
python -c "
from src.detection.cidre import AuthorCIDRE
analyzer = AuthorCIDRE('data_sample.json')
analyzer.build_author_citation_graph()
analyzer.detect_author_communities(method='connected_components')
analyzer.fit_dcSBM()
analyzer.find_excessive_citations(alpha=0.05)
groups = analyzer.detect_anomalous_groups(theta=0.1, min_citations=5)
analyzer.save_results(groups, 'author_citation_cartels.json')
print(f'Найдено аномальных групп: {len(groups)}')
"
```
3. **Иерархическая кластеризация**
```bash
python -c "
from src.detection.hierarchical import CitationCartelDetector, CartelFinder
detector = CitationCartelDetector('data_sample.json', n_papers=500)
detector.compute_factor_matrices()
detector.normalize_factors(method='robust')
detector.build_influence_matrix(threshold_percentile=99)
detector.transitive_closure()
clusters = detector.find_clusters()
detector.save_clusters_to_file('clusters_hierarchical.txt')
print(f'Найдено кластеров: {len(clusters)}')
"
```
4. **Семантическая валидация**
```bash
# Шаг 1: извлечь абстракты
python -c "
from src.data.abstract_extractor import AbstractExtractor
extractor = AbstractExtractor('data_sample.json')
extractor.extract_abstracts()
extractor.save_results('extracted_abstracts.json')
"

# Шаг 2: анализ релевантности
python -c "
from src.detection.semantic import EnhancedCitationAnalyzer
analyzer = EnhancedCitationAnalyzer(
    abstracts_file='extracted_abstracts.json',
    openalex_file='data_sample.json'
)
suspicious = analyzer.analyze_citation_relevance(threshold=0.4)
print(f'Найдено подозрительных цитирований: {len(suspicious)}')
"
```
5. **Анализ пересечений**
```bash
python -c "
from src.utils.overlap import count_works_in_suspicious_file, print_detailed_analysis
results = count_works_in_suspicious_file(
    output_json_file='author_citations_detailed.json',
    suspicious_csv_file='suspicious_citations_th_40.csv'
)
print_detailed_analysis(results)
"
```
Быстрый запуск всех методов
```bash
# Запустить все по очереди
python src/detection/threshold.py
python src/detection/cidre.py
python src/detection/hierarchical.py
python src/detection/semantic.py
```
