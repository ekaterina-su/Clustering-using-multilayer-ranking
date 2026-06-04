import json
import csv
from collections import defaultdict


def count_works_in_suspicious_file(output_json_file, suspicious_csv_file):
    """
    Считает, сколько работ из output файла присутствует в suspicious_citations файле

    Args:
        output_json_file: файл с результатами поиска цитирований (author_citations_detailed.json)
        suspicious_csv_file: файл suspicious_citations_th_40.csv

    Returns:
        dict: статистика по совпадениям
    """

    # 1. Загружаем наш output файл
    with open(output_json_file, 'r', encoding='utf-8') as f:
        output_data = json.load(f)

    # Собираем все ID статей из output файла
    our_articles = set()

    # Добавляем цитирующие статьи
    our_articles.update(output_data['article_ids']['all_citing_works'])
    # Добавляем цитируемые статьи
    our_articles.update(output_data['article_ids']['all_cited_works'])

    print(f"Всего уникальных статей в нашем output файле: {len(our_articles)}")
    print(f"Из них цитирующих: {len(output_data['article_ids']['all_citing_works'])}")
    print(f"Из них цитируемых: {len(output_data['article_ids']['all_cited_works'])}")

    # 2. Загружаем suspicious CSV файл и собираем все ID статей оттуда
    suspicious_articles = set()
    citing_in_suspicious = set()
    cited_in_suspicious = set()

    # Для подсчета пар (citing, cited) из suspicious файла
    suspicious_pairs = set()

    with open(suspicious_csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            citing = row['citing_paper_id']
            cited = row['cited_paper_id']

            suspicious_articles.add(citing)
            suspicious_articles.add(cited)
            citing_in_suspicious.add(citing)
            cited_in_suspicious.add(cited)
            suspicious_pairs.add((citing, cited))

    print(f"\nВсего уникальных статей в suspicious файле: {len(suspicious_articles)}")
    print(f"Из них цитирующих: {len(citing_in_suspicious)}")
    print(f"Из них цитируемых: {len(cited_in_suspicious)}")
    print(f"Всего пар цитирований в suspicious файле: {len(suspicious_pairs)}")

    # 3. Находим пересечения
    our_in_suspicious = our_articles.intersection(suspicious_articles)

    # Доля наших статей, которые есть в suspicious файле
    if len(our_articles) > 0:
        overlap_ratio = len(our_in_suspicious) / len(our_articles) * 100
    else:
        overlap_ratio = 0

    # 4. Проверяем, есть ли в suspicious файле конкретные пары из наших цитирований
    our_pairs = set()

    # Собираем пары из author1_citing_author2
    for cit in output_data['citations']['author1_citing_author2']:
        our_pairs.add((cit['citing_work'], cit['cited_work']))

    # Собираем пары из author2_citing_author1
    for cit in output_data['citations']['author2_citing_author1']:
        our_pairs.add((cit['citing_work'], cit['cited_work']))

    print(f"\nВсего пар цитирований в нашем файле: {len(our_pairs)}")

    # Находим пары, которые есть в suspicious файле
    pairs_in_suspicious = our_pairs.intersection(suspicious_pairs)

    if len(our_pairs) > 0:
        pairs_ratio = len(pairs_in_suspicious) / len(our_pairs) * 100
    else:
        pairs_ratio = 0

    # 5. Детальная статистика
    results = {
        'total_our_articles': len(our_articles),
        'total_suspicious_articles': len(suspicious_articles),
        'our_articles_in_suspicious': len(our_in_suspicious),
        'overlap_percentage': overlap_ratio,
        'total_our_pairs': len(our_pairs),
        'our_pairs_in_suspicious': len(pairs_in_suspicious),
        'pairs_overlap_percentage': pairs_ratio,
        'our_articles_list': sorted(list(our_articles)),
        'suspicious_articles_list': sorted(list(suspicious_articles)),
        'overlap_articles_list': sorted(list(our_in_suspicious)),
        'overlap_pairs_list': sorted([f"{p[0]} → {p[1]}" for p in pairs_in_suspicious])
    }

    return results


def print_detailed_analysis(results):
    """Подробный вывод результатов"""
    print("\n" + "=" * 70)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ СОВПАДЕНИЙ")
    print("=" * 70)

    print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
    print(f"   Наши статьи (всего): {results['total_our_articles']}")
    print(f"   Статьи в suspicious файле: {results['total_suspicious_articles']}")
    print(f"   Наших статей в suspicious файле: {results['our_articles_in_suspicious']}")
    print(f"   📈 Доля совпадений: {results['overlap_percentage']:.2f}%")

    print(f"\n🔄 ЦИТИРОВАНИЯ:")
    print(f"   Наши пары цитирований: {results['total_our_pairs']}")
    print(f"   Наших пар в suspicious файле: {results['our_pairs_in_suspicious']}")
    print(f"   📈 Доля совпадений: {results['pairs_overlap_percentage']:.2f}%")

    if results['overlap_articles_list']:
        print(f"\n✅ НАЙДЕННЫЕ СТАТЬИ (первые 10):")
        for i, article in enumerate(results['overlap_articles_list'][:10], 1):
            print(f"   {i:2}. {article}")
        if len(results['overlap_articles_list']) > 10:
            print(f"   ... и еще {len(results['overlap_articles_list']) - 10} статей")

    if results['overlap_pairs_list']:
        print(f"\n🔗 НАЙДЕННЫЕ ПАРЫ ЦИТИРОВАНИЙ (первые 10):")
        for i, pair in enumerate(results['overlap_pairs_list'][:10], 1):
            print(f"   {i:2}. {pair}")
        if len(results['overlap_pairs_list']) > 10:
            print(f"   ... и еще {len(results['overlap_pairs_list']) - 10} пар")

    return results


def save_analysis_to_file(results, output_file="overlap_analysis.json"):
    """Сохраняет результаты анализа в JSON файл"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Результаты сохранены в {output_file}")


# Запуск анализа
if __name__ == "__main__":
    # Укажите пути к вашим файлам
    our_output_file = "author_citations_detailed.json"  # файл из предыдущего шага
    suspicious_file = "suspicious_citations_th_40.csv"

    try:
        # Проводим анализ
        analysis_results = count_works_in_suspicious_file(our_output_file, suspicious_file)

        # Выводим детальный отчет
        print_detailed_analysis(analysis_results)

        # Сохраняем результаты
        save_analysis_to_file(analysis_results, "overlap_with_suspicious.json")

        # Дополнительно: считаем статистику по каждому типу статей
        print("\n" + "=" * 70)
        print("СТАТИСТИКА ПО ТИПАМ СТАТЕЙ")
        print("=" * 70)

        with open(our_output_file, 'r', encoding='utf-8') as f:
            output_data = json.load(f)

        # Проверяем цитирующие статьи
        citing_works = set(output_data['article_ids']['all_citing_works'])
        citing_in_suspicious = citing_works.intersection(analysis_results['suspicious_articles_list'])
        print(f"\n📝 Цитирующие статьи (наши): {len(citing_works)}")
        print(f"   Из них в suspicious: {len(citing_in_suspicious)}")
        print(f"   Доля: {len(citing_in_suspicious) / len(citing_works) * 100:.2f}%")

        # Проверяем цитируемые статьи
        cited_works = set(output_data['article_ids']['all_cited_works'])
        cited_in_suspicious = cited_works.intersection(analysis_results['suspicious_articles_list'])
        print(f"\n📚 Цитируемые статьи (наши): {len(cited_works)}")
        print(f"   Из них в suspicious: {len(cited_in_suspicious)}")
        print(f"   Доля: {len(cited_in_suspicious) / len(cited_works) * 100:.2f}%")

        # Проверяем статьи автор1
        author1_citing = set(output_data['article_ids']['author1_citing_works'])
        author1_in_suspicious = author1_citing.intersection(analysis_results['suspicious_articles_list'])
        if author1_citing:
            print(f"\n👤 Статьи автор1: {len(author1_citing)}")
            print(f"   Из них в suspicious: {len(author1_in_suspicious)}")
            print(f"   Доля: {len(author1_in_suspicious) / len(author1_citing) * 100:.2f}%")

        # Проверяем статьи автор2
        author2_citing = set(output_data['article_ids']['author2_citing_works'])
        author2_in_suspicious = author2_citing.intersection(analysis_results['suspicious_articles_list'])
        if author2_citing:
            print(f"\n👤 Статьи автор2: {len(author2_citing)}")
            print(f"   Из них в suspicious: {len(author2_in_suspicious)}")
            print(f"   Доля: {len(author2_in_suspicious) / len(author2_citing) * 100:.2f}%")

    except FileNotFoundError as e:
        print(f"❌ Ошибка: файл не найден - {e}")
    except json.JSONDecodeError:
        print("❌ Ошибка: не удалось прочитать JSON файл")
    except Exception as e:
        print(f"❌ Ошибка: {e}")