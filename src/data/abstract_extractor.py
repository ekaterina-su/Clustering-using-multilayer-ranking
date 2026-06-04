import json
from collections import defaultdict
from typing import Dict, List, Optional
import re


class AbstractExtractor:
    def __init__(self, json_file_path: str):
        """
        Инициализация экстрактора абстрактов

        Args:
            json_file_path: путь к файлу openalex_data.json
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.papers = self.data['results']
        self.papers_with_abstracts = []

    def reconstruct_abstract(self, inverted_index: Dict) -> str:
        """
        Восстановление абстракта из inverted_index формата OpenAlex

        OpenAlex хранит абстракты в виде inverted_index:
        {
            "word": [0, 5, 10],  # слово встречается на позициях 0, 5, 10
            "another": [1, 3],    # другое слово на позициях 1, 3
            ...
        }

        Args:
            inverted_index: словарь с инвертированным индексом

        Returns:
            восстановленный текст абстракта
        """
        if not inverted_index:
            return None

        # Находим максимальную позицию слова
        max_position = 0
        for positions in inverted_index.values():
            if positions:
                max_position = max(max_position, max(positions))

        # Создаем массив для слов
        words = [''] * (max_position + 1)

        # Заполняем массив словами на соответствующих позициях
        for word, positions in inverted_index.items():
            for pos in positions:
                if 0 <= pos < len(words):
                    words[pos] = word

        # Проверяем, все ли позиции заполнены
        missing_positions = [i for i, w in enumerate(words) if not w]
        if missing_positions:
            print(f"Предупреждение: отсутствуют слова на позициях {missing_positions}")

        # Собираем предложение
        abstract = ' '.join(words)

        # Пост-обработка: исправляем пробелы перед знаками препинания
        abstract = re.sub(r'\s+([.,;:!?)])', r'\1', abstract)
        abstract = re.sub(r'(\()\s+', r'\1', abstract)

        return abstract

    def clean_abstract(self, abstract: str) -> str:
        """
        Очистка и форматирование абстракта
        """
        if not abstract:
            return None

        # Удаляем лишние пробелы
        abstract = ' '.join(abstract.split())

        # Проверяем, начинается ли с заглавной буквы
        if abstract and abstract[0].islower():
            abstract = abstract[0].upper() + abstract[1:]

        # Добавляем точку в конце, если её нет
        if abstract and abstract[-1] not in '.!?':
            abstract += '.'

        return abstract

    def extract_abstracts(self) -> List[Dict]:
        """
        Извлечение абстрактов из всех статей
        """
        total_papers = len(self.papers)
        print(f"Всего статей в файле: {total_papers}")

        for i, paper in enumerate(self.papers, 1):
            paper_id = paper.get('id', 'N/A')
            title = paper.get('title', 'Без названия')

            # Получаем inverted_index
            inverted_index = paper.get('abstract_inverted_index')

            # Получаем авторов
            authors = []
            for authorship in paper.get('authorships', []):
                author = authorship.get('author', {})
                author_name = author.get('display_name', 'Unknown')
                authors.append(author_name)

            # Восстанавливаем абстракт
            abstract_text = None
            if inverted_index:
                abstract_text = self.reconstruct_abstract(inverted_index)
                abstract_text = self.clean_abstract(abstract_text)

            paper_info = {
                'id': paper_id,
                'doi': paper.get('doi'),
                'title': title,
                'publication_year': paper.get('publication_year'),
                'publication_date': paper.get('publication_date'),
                'authors': authors,
                'authors_count': len(authors),
                'cited_by_count': paper.get('cited_by_count', 0),
                'has_abstract': abstract_text is not None,
                'abstract': abstract_text,
                'abstract_length': len(abstract_text) if abstract_text else 0,
                'abstract_word_count': len(abstract_text.split()) if abstract_text else 0
            }

            self.papers_with_abstracts.append(paper_info)

            # Прогресс
            if i % 10 == 0 or i == total_papers:
                print(f"Обработано {i}/{total_papers} статей...")

        return self.papers_with_abstracts

    def save_results(self, output_file: str = 'extracted_abstracts.json'):
        """
        Сохранение результатов в JSON файл
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_papers': len(self.papers_with_abstracts),
                'papers_with_abstracts': sum(1 for p in self.papers_with_abstracts if p['has_abstract']),
                'papers_without_abstracts': sum(1 for p in self.papers_with_abstracts if not p['has_abstract']),
                'papers': self.papers_with_abstracts
            }, f, ensure_ascii=False, indent=2)

        print(f"\nРезультаты сохранены в {output_file}")

    def save_abstracts_only(self, output_file: str = 'abstracts_only.txt'):
        """
        Сохранение только текстов абстрактов в текстовый файл
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for paper in self.papers_with_abstracts:
                if paper['has_abstract']:
                    f.write(f"DOI: {paper['doi']}\n")
                    f.write(f"Title: {paper['title']}\n")
                    f.write(f"Authors: {', '.join(paper['authors'])}\n")
                    f.write(f"Year: {paper['publication_year']}\n")
                    f.write("-" * 80 + "\n")
                    f.write(paper['abstract'])
                    f.write("\n\n" + "=" * 80 + "\n\n")

        print(f"Тексты абстрактов сохранены в {output_file}")

    def print_statistics(self):
        """
        Вывод статистики по абстрактам
        """
        total = len(self.papers_with_abstracts)
        with_abstract = sum(1 for p in self.papers_with_abstracts if p['has_abstract'])
        without_abstract = total - with_abstract

        print("\n" + "=" * 60)
        print("СТАТИСТИКА ПО АБСТРАКТАМ")
        print("=" * 60)
        print(f"Всего статей: {total}")
        print(f"С абстрактами: {with_abstract} ({with_abstract / total * 100:.1f}%)")
        print(f"Без абстрактов: {without_abstract} ({without_abstract / total * 100:.1f}%)")

        if with_abstract > 0:
            # Статистика по длине абстрактов
            lengths = [p['abstract_length'] for p in self.papers_with_abstracts if p['has_abstract']]
            word_counts = [p['abstract_word_count'] for p in self.papers_with_abstracts if p['has_abstract']]

            print(f"\nСтатистика длины абстрактов:")
            print(f"  Средняя длина (символы): {sum(lengths) / len(lengths):.0f}")
            print(f"  Минимальная длина: {min(lengths)}")
            print(f"  Максимальная длина: {max(lengths)}")
            print(f"  Среднее количество слов: {sum(word_counts) / len(word_counts):.1f}")

            # Распределение по годам
            years = {}
            for p in self.papers_with_abstracts:
                year = p.get('publication_year')
                if year and p['has_abstract']:
                    years[year] = years.get(year, 0) + 1

            if years:
                print(f"\nАбстракты по годам:")
                for year in sorted(years.keys()):
                    print(f"  {year}: {years[year]} статей")

    def print_sample_abstracts(self, n: int = 5):
        """
        Вывод примеров абстрактов
        """
        papers_with_abs = [p for p in self.papers_with_abstracts if p['has_abstract']]

        print(f"\n" + "=" * 60)
        print(f"ПРИМЕРЫ АБСТРАКТОВ (первые {min(n, len(papers_with_abs))})")
        print("=" * 60)

        for i, paper in enumerate(papers_with_abs[:n], 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   Авторы: {', '.join(paper['authors'][:3])}" +
                  (f" и др." if len(paper['authors']) > 3 else ""))
            print(f"   DOI: {paper['doi']}")
            print(f"   Год: {paper['publication_year']}")
            print(f"   Длина абстракта: {paper['abstract_length']} символов, {paper['abstract_word_count']} слов")
            print("-" * 40)

            # Выводим абстракт с разбивкой по строкам для читаемости
            abstract = paper['abstract']
            words = abstract.split()
            lines = []
            current_line = []

            for word in words:
                current_line.append(word)
                if len(' '.join(current_line)) > 80:
                    lines.append(' '.join(current_line[:-1]))
                    current_line = [current_line[-1]]

            if current_line:
                lines.append(' '.join(current_line))

            for line in lines:
                print(line)

    def export_to_csv(self, output_file: str = 'abstracts_data.csv'):
        """
        Экспорт данных в CSV файл
        """
        import csv

        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)

            # Заголовки
            writer.writerow([
                'DOI', 'Title', 'Publication Year', 'Authors',
                'Cited By Count', 'Has Abstract', 'Abstract Length',
                'Abstract Word Count', 'Abstract Preview'
            ])

            # Данные
            for paper in self.papers_with_abstracts:
                abstract_preview = ''
                if paper['has_abstract']:
                    abstract_preview = paper['abstract'][:200] + '...' if len(paper['abstract']) > 200 else paper[
                        'abstract']

                writer.writerow([
                    paper['doi'],
                    paper['title'],
                    paper['publication_year'],
                    ', '.join(paper['authors'][:5]) + ('...' if len(paper['authors']) > 5 else ''),
                    paper['cited_by_count'],
                    'Да' if paper['has_abstract'] else 'Нет',
                    paper['abstract_length'],
                    paper['abstract_word_count'],
                    abstract_preview
                ])

        print(f"\nДанные экспортированы в {output_file}")


def main():
    # Путь к вашему JSON файлу
    json_file = 'openalex_data.json'

    print("=" * 60)
    print("ИЗВЛЕЧЕНИЕ АБСТРАКТОВ ИЗ OPENALEX DATA")
    print("=" * 60)

    # Создаем экстрактор
    extractor = AbstractExtractor(json_file)

    # Извлекаем абстракты
    papers = extractor.extract_abstracts()

    # Выводим статистику
    extractor.print_statistics()

    # Выводим примеры
    extractor.print_sample_abstracts(n=5)

    # Сохраняем результаты
    extractor.save_results('extracted_abstracts.json')
    extractor.save_abstracts_only('abstracts_only.txt')
    extractor.export_to_csv('abstracts_data.csv')

    print("\n" + "=" * 60)
    print("ГОТОВО! Все файлы сохранены:")
    print("  - extracted_abstracts.json (полные данные в JSON)")
    print("  - abstracts_only.txt (только тексты абстрактов)")
    print("  - abstracts_data.csv (данные в формате CSV)")
    print("=" * 60)


if __name__ == "__main__":
    main()