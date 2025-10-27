import argparse
import sys
import os
import json
from typing import Dict, Any


class BaseDependencyParser:
    def parse_dependencies(self, content: str) -> Dict[str, str]:
        raise NotImplementedError


class RequirementsParser(BaseDependencyParser):

    def parse_dependencies(self, content: str) -> Dict[str, str]:
        dependencies = {}

        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            if '==' in line:
                parts = line.split('==', 1)
                package = parts[0].strip()
                version = parts[1].strip()
            elif '>=' in line:
                parts = line.split('>=', 1)
                package = parts[0].strip()
                version = f">={parts[1].strip()}"
            elif '@' in line:
                package = line.split('@')[0].strip()
            else:
                package = line
                version = "unspecified"

            if package:
                dependencies[package] = version

        return dependencies


class PackageJsonParser(BaseDependencyParser):

    def parse_dependencies(self, content: str) -> Dict[str, str]:
        try:
            data = json.loads(content)
            dependencies = {}

            if 'dependencies' in data:
                dependencies.update(data['dependencies'])

            if 'devDependencies' in data:
                for dep, version in data['devDependencies'].items():
                    dependencies[f"{dep} (dev)"] = version

            return dependencies

        except json.JSONDecodeError as e:
            raise ValueError(f"Ошибка парсинга JSON: {e}")


class PomXmlParser(BaseDependencyParser):

    def parse_dependencies(self, content: str) -> Dict[str, str]:
        dependencies = {}

        lines = content.split('\n')
        in_dependencies = False

        for line in lines:
            line = line.strip()

            if '<dependencies>' in line:
                in_dependencies = True
                continue
            elif '</dependencies>' in line:
                in_dependencies = False
                continue

            if in_dependencies and '<groupId>' in line:
                group_id = self._extract_value(line, 'groupId')
                artifact_id = None
                version = None

                for next_line in lines[lines.index(line) + 1:]:
                    if '<artifactId>' in next_line:
                        artifact_id = self._extract_value(next_line, 'artifactId')
                    elif '<version>' in next_line:
                        version = self._extract_value(next_line, 'version')
                    elif '</dependency>' in next_line:
                        break

                if group_id and artifact_id:
                    package_name = f"{group_id}:{artifact_id}"
                    dependencies[package_name] = version or "unspecified"

        return dependencies

    def _extract_value(self, line: str, tag: str) -> str:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"

        start_idx = line.find(start_tag) + len(start_tag)
        end_idx = line.find(end_tag)

        if start_idx >= len(start_tag) and end_idx > start_idx:
            return line[start_idx:end_idx].strip()

        return ""


class DependencyVisualizer:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.parsers = {
            'requirements.txt': RequirementsParser(),
            'package.json': PackageJsonParser(),
            'pom.xml': PomXmlParser()
        }

    def detect_file_type(self, file_path: str) -> str:
        filename = os.path.basename(file_path).lower()

        if filename == 'requirements.txt':
            return 'requirements.txt'
        elif filename == 'package.json':
            return 'package.json'
        elif filename == 'pom.xml':
            return 'pom.xml'
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(1000)  # Читаем первые 1000 символов

                if '"dependencies"' in content or '"devDependencies"' in content:
                    return 'package.json'
                elif '<dependencies>' in content:
                    return 'pom.xml'
                else:
                    return 'requirements.txt'
            except:
                return 'requirements.txt'

    def parse_dependencies_from_file(self, file_path: str) -> Dict[str, str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            file_type = self.detect_file_type(file_path)
            parser = self.parsers.get(file_type)

            if not parser:
                raise ValueError(f"Неподдерживаемый тип файла: {file_type}")

            return parser.parse_dependencies(content)

        except FileNotFoundError:
            raise ValueError(f"Файл не найден: {file_path}")
        except Exception as e:
            raise ValueError(f"Ошибка при чтении файла {file_path}: {e}")

    def filter_dependencies(self, dependencies: Dict[str, str]) -> Dict[str, str]:
        if not self.config['filter_substring']:
            return dependencies

        filtered = {}
        for package, version in dependencies.items():
            if self.config['filter_substring'].lower() in package.lower():
                filtered[package] = version

        return filtered

    def validate_parameters(self) -> bool:

        if not self.config['package_name']:
            print("Ошибка: Имя пакета не может быть пустым")
            return False

        if not self.config['repository']:
            print("Ошибка: Путь к файлу зависимостей не может быть пустым")
            return False

        if self.config['test_mode']:
            if not os.path.exists(self.config['repository']):
                print(f"Ошибка: Файл не найден: {self.config['repository']}")
                return False

        if not self.config['output_file']:
            print("Ошибка: Имя выходного файла не может быть пустым")
            return False

        valid_extensions = ['.png', '.jpg', '.jpeg', '.svg', '.pdf']
        if not any(self.config['output_file'].lower().endswith(ext) for ext in valid_extensions):
            print("Ошибка: Неподдерживаемое расширение файла. Используйте: .png, .jpg, .jpeg, .svg, .pdf")
            return False

        return True

    def display_config(self):
        print("\n    Конфигурация приложения")
        for key, value in self.config.items():
            print(f"{key}: {value}")

    def analyze_dependencies(self):
        print(f"Анализ зависимостей для пакета: {self.config['package_name']}")
        print(f"Источник: {self.config['repository']}")

        try:
            dependencies = self.parse_dependencies_from_file(self.config['repository'])

            if self.config['filter_substring']:
                dependencies = self.filter_dependencies(dependencies)
                print(f"Применен фильтр: '{self.config['filter_substring']}'")

            print(f"\nНайдено зависимостей: {len(dependencies)}")
            if dependencies:
                print("\nСписок зависимостей:")
                for package, version in dependencies.items():
                    print(f"  - {package}: {version}")
            else:
                print("Зависимости не найдены или отфильтрованы")

            print(f"\nРезультат будет сохранен в: {self.config['output_file']}")

        except ValueError as e:
            print(f"Ошибка анализа: {e}")
            return False

        return True


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Инструмент визуализации графа зависимостей пакетов',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Примеры использования:
  python main.py --package myapp --repo requirements.txt --output deps.png
  python main.py --package myapp --repo package.json --test-mode --filter "react"
  python main.py --package myproject --repo pom.xml --output graph.svg --filter "spring"
        '''
    )

    parser.add_argument(
        '--package', '--package-name',
        dest='package_name',
        required=True,
        help='Имя анализируемого пакета'
    )

    parser.add_argument(
        '--repo', '--repository',
        dest='repository',
        required=True,
        help='Путь к файлу зависимостей (requirements.txt, package.json, pom.xml)'
    )

    parser.add_argument(
        '--output', '--output-file',
        dest='output_file',
        default='dependency_graph.png',
        help='Имя сгенерированного файла с изображением графа'
    )

    parser.add_argument(
        '--test-mode',
        dest='test_mode',
        action='store_true',
        default=False,
        help='Режим работы с тестовым репозиторием'
    )

    parser.add_argument(
        '--filter',
        dest='filter_substring',
        default='',
        help='Подстрока для фильтрации пакетов'
    )

    return parser.parse_args()


def main():
    try:
        args = parse_arguments()

        config = {
            'package_name': args.package_name,
            'repository': args.repository,
            'output_file': args.output_file,
            'test_mode': args.test_mode,
            'filter_substring': args.filter_substring
        }

        visualizer = DependencyVisualizer(config)

        if not visualizer.validate_parameters():
            sys.exit(1)

        visualizer.display_config()

        if not visualizer.analyze_dependencies():
            sys.exit(1)

    except argparse.ArgumentError as e:
        print(f"Ошибка в аргументах командной строки: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nПрограмма прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()