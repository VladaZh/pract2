import argparse
import sys
import os
import json
import requests
from typing import Dict, Any, List, Set
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx


class NPMPackageParser:
    def __init__(self):
        self.npm_registry_url = "https://registry.npmjs.org"

    def get_package_info(self, package_name: str) -> Dict[str, Any]:
        try:
            url = f"{self.npm_registry_url}/{package_name}"
            response = requests.get(url, timeout=10)

            if response.status_code == 404:
                raise ValueError(f"Пакет '{package_name}' не найден в npm registry")
            elif response.status_code != 200:
                raise ValueError(f"Ошибка npm registry: {response.status_code}")

            return response.json()

        except requests.exceptions.RequestException as e:
            raise ValueError(f"Ошибка подключения к npm registry: {e}")

    def get_dependencies(self, package_name: str, version: str = "latest") -> Dict[str, str]:
        package_info = self.get_package_info(package_name)

        if version == "latest":
            version = package_info.get('dist-tags', {}).get('latest', 'latest')

        version_info = package_info.get('versions', {}).get(version)
        if not version_info:
            raise ValueError(f"Версия '{version}' не найдена для пакета '{package_name}'")

        dependencies = {}

        deps = version_info.get('dependencies', {})
        for dep, version_spec in deps.items():
            dependencies[dep] = version_spec

        dev_deps = version_info.get('devDependencies', {})
        for dep, version_spec in dev_deps.items():
            dependencies[f"{dep} (dev)"] = version_spec

        peer_deps = version_info.get('peerDependencies', {})
        for dep, version_spec in peer_deps.items():
            dependencies[f"{dep} (peer)"] = version_spec

        return dependencies


class URLRepositoryParser:

    def __init__(self):
        self.npm_parser = NPMPackageParser()

    def parse_from_url(self, url: str, package_name: str) -> Dict[str, str]:

        if 'npmjs.org' in url or 'registry.npmjs.org' in url:
            return self.npm_parser.get_dependencies(package_name)

        elif 'github.com' in url:
            return self._parse_github_repo(url, package_name)

        else:
            raise ValueError(f"Неподдерживаемый URL репозитория: {url}")

    def _parse_github_repo(self, github_url: str, package_name: str) -> Dict[str, str]:
        try:
            # Преобразуем GitHub URL в raw content URL
            if 'github.com' in github_url and '/blob/' in github_url:
                raw_url = github_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
            else:
                repo_path = github_url.replace('https://github.com/', '')
                raw_url = f"https://raw.githubusercontent.com/{repo_path}/main/package.json"

            response = requests.get(raw_url, timeout=10)

            if response.status_code == 404:
                raw_url = f"https://raw.githubusercontent.com/{repo_path}/master/package.json"
                response = requests.get(raw_url, timeout=10)

            if response.status_code != 200:
                raise ValueError(f"Не удалось получить package.json из репозитория")

            package_data = response.json()
            return self._extract_dependencies_from_package_json(package_data)

        except requests.exceptions.RequestException as e:
            raise ValueError(f"Ошибка подключения к GitHub: {e}")
        except json.JSONDecodeError:
            raise ValueError("Некорректный JSON в package.json")

    def _extract_dependencies_from_package_json(self, package_data: Dict) -> Dict[str, str]:
        dependencies = {}

        deps = package_data.get('dependencies', {})
        for dep, version in deps.items():
            dependencies[dep] = version

        dev_deps = package_data.get('devDependencies', {})
        for dep, version in dev_deps.items():
            dependencies[f"{dep} (dev)"] = version

        return dependencies


class TestRepositoryParser:
    """Парсер для тестового репозитория с пакетами в виде больших латинских букв"""

    def __init__(self):
        self.test_graphs = self._create_test_graphs()

    def _create_test_graphs(self) -> Dict[str, Dict[str, List[str]]]:
        """Создает тестовые графы зависимостей"""
        return {
            # Простой линейный граф: A -> B -> C -> D
            "linear": {
                "A": ["B"],
                "B": ["C"],
                "C": ["D"],
                "D": []
            },
            # Граф с ветвлением: A -> [B, C], B -> D, C -> D
            "branching": {
                "A": ["B", "C"],
                "B": ["D"],
                "C": ["D"],
                "D": []
            },
            # Циклический граф: A -> B -> C -> A
            "cyclic": {
                "A": ["B"],
                "B": ["C"],
                "C": ["A"]
            },
            # Сложный граф с несколькими циклами
            "complex": {
                "A": ["B", "C"],
                "B": ["D", "E"],
                "C": ["F"],
                "D": ["G", "H"],
                "E": ["H", "I"],
                "F": ["I", "J"],
                "G": ["K"],
                "H": ["K", "L"],
                "I": ["L", "M"],
                "J": ["M"],
                "K": ["N"],
                "L": ["N"],
                "M": ["N"],
                "N": []
            },
            # Граф с самозависимостью
            "self_cyclic": {
                "A": ["A", "B"],
                "B": ["C"],
                "C": []
            }
        }

    def parse_test_file(self, file_path: str) -> Dict[str, List[str]]:
        """Парсит тестовый файл с описанием графа"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            # Поддерживаем два формата: JSON и простой текстовый
            if content.startswith('{'):
                # JSON формат
                graph_data = json.loads(content)
                # Преобразуем в нужный формат (списки зависимостей)
                result = {}
                for package, deps in graph_data.items():
                    if isinstance(deps, list):
                        result[package] = deps
                    elif isinstance(deps, dict):
                        result[package] = list(deps.keys())
                    else:
                        result[package] = []
                return result
            else:
                # Простой текстовый формат: каждая строка "ПАКЕТ: ЗАВИСИМОСТИ"
                graph = defaultdict(list)
                for line in content.split('\n'):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    if ':' in line:
                        package, deps_str = line.split(':', 1)
                        package = package.strip()
                        deps = [dep.strip() for dep in deps_str.split(',') if dep.strip()]
                        graph[package] = deps
                    else:
                        # Если нет двоеточия, считаем пакетом без зависимостей
                        graph[line.strip()] = []

                return dict(graph)

        except Exception as e:
            raise ValueError(f"Ошибка чтения тестового файла: {e}")


class DependencyGraph:
    """Класс для работы с графом зависимостей"""

    def __init__(self):
        self.graph = defaultdict(list)
        self.visited = set()
        self.recursion_stack = set()
        self.cycles = []
        self.filter_substring = ""

    def add_dependency(self, package: str, dependency: str):
        """Добавляет зависимость в граф"""
        if dependency and dependency not in self.graph[package]:
            self.graph[package].append(dependency)

    def should_skip_package(self, package: str) -> bool:
        """Проверяет, нужно ли пропустить пакет согласно фильтру"""
        if not self.filter_substring:
            return False
        return self.filter_substring.lower() in package.lower()

    def dfs_with_cycles_detection(self, node: str, path: List[str]):
        """Рекурсивный DFS с обнаружением циклов"""
        if self.should_skip_package(node):
            return

        if node in self.recursion_stack:
            # Найден цикл
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            if cycle not in self.cycles:
                self.cycles.append(cycle)
            return

        if node in self.visited:
            return

        self.visited.add(node)
        self.recursion_stack.add(node)
        path.append(node)

        # Рекурсивно обходим зависимости
        for neighbor in self.graph.get(node, []):
            if not self.should_skip_package(neighbor):
                self.dfs_with_cycles_detection(neighbor, path.copy())

        self.recursion_stack.remove(node)
        path.pop()

    def find_all_cycles(self) -> List[List[str]]:
        """Находит все циклы в графе"""
        self.visited.clear()
        self.recursion_stack.clear()
        self.cycles.clear()

        for node in list(self.graph.keys()):
            if node not in self.visited and not self.should_skip_package(node):
                self.dfs_with_cycles_detection(node, [])

        return self.cycles

    def get_transitive_dependencies(self, start_package: str) -> Set[str]:
        """Получает все транзитивные зависимости для пакета"""
        if self.should_skip_package(start_package):
            return set()

        visited = set()

        def dfs_transitive(node: str):
            if self.should_skip_package(node) or node in visited:
                return
            visited.add(node)
            for neighbor in self.graph.get(node, []):
                if not self.should_skip_package(neighbor):
                    dfs_transitive(neighbor)

        dfs_transitive(start_package)
        visited.discard(start_package)  # Убираем стартовый пакет из результата
        return visited

    def build_complete_dependency_graph(self, start_packages: List[str]) -> Dict[str, Set[str]]:
        """Строит полный граф зависимостей для начальных пакетов"""
        result = {}
        for package in start_packages:
            if not self.should_skip_package(package):
                result[package] = self.get_transitive_dependencies(package)
        return result

    def print_graph(self):
        """Выводит граф для отладки"""
        print("\nСТРУКТУРА ГРАФА:")
        if not self.graph:
            print("  Граф пуст")
            return

        for package, deps in sorted(self.graph.items()):
            if deps:
                print(f"  {package} -> {', '.join(sorted(deps))}")
            else:
                print(f"  {package} -> (нет зависимостей)")


class DependencyVisualizer:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.url_parser = URLRepositoryParser()
        self.npm_parser = NPMPackageParser()
        self.test_parser = TestRepositoryParser()
        self.dependency_graph = DependencyGraph()

    def is_url(self, repository: str) -> bool:
        return repository.startswith(('http://', 'https://'))

    def validate_parameters(self) -> bool:

        if not self.config['package_name']:
            print("Ошибка: Имя пакета не может быть пустым")
            return False

        if not self.config['repository']:
            print("Ошибка: URL репозитория или путь к файлу не может быть пустым")
            return False

        if self.config['test_mode'] and not self.is_url(self.config['repository']):
            if not os.path.exists(self.config['repository']):
                print(f"Ошибка: Файл не найден: {self.config['repository']}")
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
            dependencies_data = {}

            if self.config['test_mode']:
                print("Режим: Тестовый репозиторий")
                if self.is_url(self.config['repository']):
                    # Для тестового режима с URL используем предопределенные графы
                    graph_name = self.config['repository'].split('/')[-1]
                    dependencies_data = self.test_parser.test_graphs.get(graph_name, {})
                    print(f"Используется встроенный граф: {graph_name}")
                else:
                    # Чтение из файла
                    dependencies_data = self.test_parser.parse_test_file(self.config['repository'])
                    print(f"Прочитан граф из файла: {self.config['repository']}")
            else:
                if self.is_url(self.config['repository']):
                    print("Режим: URL репозитория")
                    dependencies_data = self.url_parser.parse_from_url(
                        self.config['repository'],
                        self.config['package_name']
                    )
                else:
                    print("Режим: Локальный файл")
                    dependencies_data = self.npm_parser.get_dependencies(self.config['package_name'])

            # Строим граф зависимостей
            self.build_dependency_graph(dependencies_data)

            # Применяем фильтр если указан
            if self.config['filter_substring']:
                self.dependency_graph.filter_substring = self.config['filter_substring']
                print(f"Применен фильтр: '{self.config['filter_substring']}'")

            # Анализируем граф
            self.analyze_dependency_graph()

            # Визуализируем граф
            self.visualize_dependency_graph()

            print(f"\nРезультат сохранен в: {self.config['output_file']}")

        except ValueError as e:
            print(f"Ошибка анализа: {e}")
            return False

        return True

    def build_dependency_graph(self, dependencies_data: Dict[str, Any]):
        """Строит граф зависимостей из полученных данных"""
        if self.config['test_mode']:
            # Для тестовых данных (уже готовый граф)
            print("Построение графа из тестовых данных...")
            for package, deps in dependencies_data.items():
                for dep in deps:
                    if dep:  # проверяем, что зависимость не пустая
                        self.dependency_graph.add_dependency(package, dep)
        else:
            # Для реальных npm зависимостей
            print("Построение графа из npm зависимостей...")
            for package, version in dependencies_data.items():
                self.dependency_graph.add_dependency(self.config['package_name'], package)

    def analyze_dependency_graph(self):
        """Анализирует граф зависимостей"""
        # Выводим структуру графа для отладки
        self.dependency_graph.print_graph()

        print(f"\n{'=' * 50}")
        print("АНАЛИЗ ГРАФА ЗАВИСИМОСТЕЙ")
        print(f"{'=' * 50}")

        # Поиск циклов
        cycles = self.dependency_graph.find_all_cycles()
        if cycles:
            print(f"\nОбнаружено циклических зависимостей: {len(cycles)}")
            for i, cycle in enumerate(cycles, 1):
                print(f"Цикл {i}: {' -> '.join(cycle)}")
        else:
            print("\nЦиклические зависимости не обнаружены")

        # Транзитивные зависимости
        transitive_deps = self.dependency_graph.get_transitive_dependencies(self.config['package_name'])
        print(f"\nТранзитивные зависимости для '{self.config['package_name']}': {len(transitive_deps)}")
        if transitive_deps:
            for dep in sorted(transitive_deps):
                print(f"  - {dep}")
        else:
            print("  (нет транзитивных зависимостей)")

        # Полный граф зависимостей
        complete_graph = self.dependency_graph.build_complete_dependency_graph([self.config['package_name']])
        print(f"\nПолный граф зависимостей построен для {len(complete_graph)} пакетов")

    def visualize_dependency_graph(self):
        """Визуализирует граф зависимостей"""
        try:
            G = nx.DiGraph()

            # Добавляем узлы и ребра
            for package, dependencies in self.dependency_graph.graph.items():
                if self.dependency_graph.should_skip_package(package):
                    continue
                G.add_node(package)
                for dep in dependencies:
                    if not self.dependency_graph.should_skip_package(dep):
                        G.add_edge(package, dep)

            if len(G.nodes) == 0:
                print("Нет узлов для визуализации после применения фильтра")
                return

            # Создаем визуализацию
            plt.figure(figsize=(14, 10))

            # Используем разные алгоритмы размещения для лучшего отображения
            try:
                if len(G.nodes) <= 8:
                    pos = nx.spring_layout(G, k=3, iterations=100)
                elif len(G.nodes) <= 15:
                    pos = nx.spring_layout(G, k=2, iterations=200)
                else:
                    pos = nx.spring_layout(G, k=1.5, iterations=300)
            except:
                # Fallback если spring_layout не работает
                pos = nx.random_layout(G)

            # Определяем цвета узлов в зависимости от их роли
            node_colors = []
            for node in G.nodes():
                if node == self.config['package_name']:
                    node_colors.append('lightcoral')  # Красный для корневого пакета
                elif G.in_degree(node) == 0 and G.out_degree(node) > 0:
                    node_colors.append('lightgreen')  # Зеленый для листьев
                else:
                    node_colors.append('lightblue')  # Синий для остальных

            # Рисуем граф
            nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                   node_size=1200, alpha=0.9, linewidths=2,
                                   edgecolors='black')

            # Рисуем ребра с разными стилями для циклов
            edge_colors = []
            for edge in G.edges():
                # Проверяем, является ли ребро частью цикла
                if self.is_edge_in_cycle(edge[0], edge[1]):
                    edge_colors.append('red')
                else:
                    edge_colors.append('gray')

            nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                                   arrows=True, arrowsize=25, alpha=0.8,
                                   arrowstyle='->', width=2.5,
                                   connectionstyle='arc3,rad=0.1')

            nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

            # Добавляем легенду
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightcoral', label=f"Корневой пакет ({self.config['package_name']})"),
                Patch(facecolor='lightblue', label='Промежуточные пакеты'),
                Patch(facecolor='lightgreen', label='Конечные пакеты'),
            ]
            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

            plt.title(f"Граф зависимостей для '{self.config['package_name']}'\n"
                      f"Всего узлов: {len(G.nodes)}, связей: {len(G.edges)}\n"
                      f"Фильтр: '{self.config['filter_substring']}'",
                      size=14, pad=20)
            plt.axis('off')
            plt.tight_layout()

            # Сохраняем изображение
            plt.savefig(self.config['output_file'], dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Визуализация графа сохранена в {self.config['output_file']}")

        except Exception as e:
            print(f"Ошибка при визуализации графа: {e}")
            import traceback
            traceback.print_exc()

    def is_edge_in_cycle(self, source: str, target: str) -> bool:
        """Проверяет, является ли ребро частью цикла"""
        for cycle in self.dependency_graph.cycles:
            for i in range(len(cycle) - 1):
                if cycle[i] == source and cycle[i + 1] == target:
                    return True
            # Проверяем замыкание цикла
            if cycle[-1] == source and cycle[0] == target:
                return True
        return False

    def filter_dependencies(self, dependencies: Dict[str, str]) -> Dict[str, str]:
        if not self.config['filter_substring']:
            return dependencies

        filtered = {}
        for package, version in dependencies.items():
            if self.config['filter_substring'].lower() in package.lower():
                filtered[package] = version

        return filtered

    def display_direct_dependencies(self, dependencies: Dict[str, str]):
        print(f"\n Прямые зависимости пакета '{self.config['package_name'].upper()}' ")

        if not dependencies:
            print("Прямые зависимости не найдены")
            return

        print(f"Найдено прямых зависимостей: {len(dependencies)}")

        regular_deps = {k: v for k, v in dependencies.items() if '(dev)' not in k and '(peer)' not in k}
        dev_deps = {k: v for k, v in dependencies.items() if '(dev)' in k}
        peer_deps = {k: v for k, v in dependencies.items() if '(peer)' in k}

        if regular_deps:
            print("\nОсновные зависимости:")
            for package, version in sorted(regular_deps.items()):
                print(f"   {package}: {version}")

        if dev_deps:
            print("\nЗависимости разработки:")
            for package, version in sorted(dev_deps.items()):
                print(f"   {package}: {version}")

        if peer_deps:
            print("\nPeer зависимости:")
            for package, version in sorted(peer_deps.items()):
                print(f"   {package}: {version}")

        print("")


def create_test_files():
    """Создает тестовые файлы для демонстрации"""
    test_files = {
        "test_linear.txt": """A: B
B: C
C: D
D:""",

        "test_branching.txt": """A: B, C
B: D
C: D, E
D: F
E: F
F:""",

        "test_cyclic.json": {
            "A": ["B"],
            "B": ["C"],
            "C": ["A", "D"],
            "D": []
        },

        "test_complex.txt": """A: B, C
B: D, E
C: F, G
D: H
E: H, I
F: I, J
G: K
H: L
I: L, M
J: N
K: N
L: O
M: O
N: O
O:"""
    }

    for filename, content in test_files.items():
        try:
            if filename.endswith('.json'):
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(content, f, indent=2)
            else:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
            print(f"Создан тестовый файл: {filename}")
        except Exception as e:
            print(f"Ошибка создания файла {filename}: {e}")


def demonstrate_functionality():
    """Демонстрирует функциональность на различных тестовых случаях"""
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ ФУНКЦИОНАЛЬНОСТИ")
    print("=" * 60)

    # Создаем тестовые файлы
    create_test_files()

    test_cases = [
        {
            "name": "Линейный граф без фильтра",
            "package": "A",
            "repo": "test_linear.txt",
            "filter": "",
            "test_mode": True
        },
        {
            "name": "Ветвящийся граф с фильтром 'D'",
            "package": "A",
            "repo": "test_branching.txt",
            "filter": "D",
            "test_mode": True
        },
        {
            "name": "Циклический граф",
            "package": "A",
            "repo": "test_cyclic.json",
            "filter": "",
            "test_mode": True
        },
        {
            "name": "Сложный граф с фильтром 'L'",
            "package": "A",
            "repo": "test_complex.txt",
            "filter": "L",
            "test_mode": True
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'─' * 50}")
        print(f"ТЕСТ {i}: {test_case['name']}")
        print(f"{'─' * 50}")

        config = {
            'package_name': test_case['package'],
            'repository': test_case['repo'],
            'output_file': f'test_output_{i}.png',
            'test_mode': test_case['test_mode'],
            'filter_substring': test_case['filter']
        }

        visualizer = DependencyVisualizer(config)
        visualizer.analyze_dependencies()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Инструмент визуализации графа зависимостей пакетов',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '--package', '--package-name',
        dest='package_name',
        required=False,
        help='Имя анализируемого npm пакета'
    )

    parser.add_argument(
        '--repo', '--repository',
        dest='repository',
        required=False,
        help='URL npm registry или GitHub репозитория, либо путь к локальному файлу'
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

    parser.add_argument(
        '--demo',
        dest='demo_mode',
        action='store_true',
        default=False,
        help='Запуск демонстрации на тестовых данных'
    )

    return parser.parse_args()


def main():
    try:
        args = parse_arguments()

        if args.demo_mode:
            demonstrate_functionality()
            return

        # Проверяем, что для не-demo режима указаны обязательные аргументы
        if not args.package_name or not args.repository:
            print("Ошибка: для обычного режима необходимо указать --package и --repo")
            print("Используйте --demo для демонстрационного режима")
            sys.exit(1)

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