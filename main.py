import argparse
import sys
import os
import json
import requests
from typing import Dict, Any, List, Set
from collections import defaultdict
import graphviz


class NPMPackageParser: # формат пакетов JavaScript (npm)
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


class URLRepositoryParser: # информация о прямых зависимостях через URL-адрес репозитория

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


class TestRepositoryParser: # режим тестирования

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

    def should_skip_package(self, package: str) -> bool: # Не учитывает при анализе пакеты, имя которых содержит заданную пользователем подстроку
       # фильтр
        if not self.filter_substring:
            return False
        return self.filter_substring.lower() in package.lower()

    def get_filtered_graph(self) -> Dict[str, List[str]]:
        """Возвращает отфильтрованную версию графа"""
        filtered_graph = {}
        for package, deps in self.graph.items():
            if self.should_skip_package(package):
                continue
            filtered_deps = [dep for dep in deps if not self.should_skip_package(dep)]
            filtered_graph[package] = filtered_deps
        return filtered_graph

    def dfs_with_cycles_detection(self, node: str, path: List[str]): # рекурсивный алгоритм DFS с рекурсией с обнаружением циклов
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

    def find_all_cycles(self) -> List[List[str]]: # обработка случаев наличия циклических зависимостей
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
        result = {}
        for package in start_packages:
            if not self.should_skip_package(package):
                result[package] = self.get_transitive_dependencies(package)
        return result

    def print_graph(self):
        print("\nСТРУКТУРА ГРАФА (после фильтрации):")
        filtered_graph = self.get_filtered_graph()

        if not filtered_graph:
            print("  Граф пуст после применения фильтра")
            return

        for package, deps in sorted(filtered_graph.items()):
            if deps:
                print(f"  {package} -> {', '.join(sorted(deps))}")
            else:
                print(f"  {package} -> (нет зависимостей)")

    def get_load_order(self, start_package: str) -> List[str]:
        """Возвращает порядок загрузки зависимостей"""
        if self.should_skip_package(start_package):
            return []

        visited = set()
        load_order = []

        def dfs_topological(node: str):
            if self.should_skip_package(node) or node in visited:
                return
            visited.add(node)

            # Сначала посещаем всех соседей (зависимости)
            for neighbor in self.graph.get(node, []):
                if not self.should_skip_package(neighbor):
                    dfs_topological(neighbor)

            # Затем добавляем текущий узел в порядок загрузки
            load_order.append(node)

        dfs_topological(start_package)
        return load_order

    def get_detailed_load_order(self, start_package: str) -> List[Dict[str, Any]]:
        """Возвращает детализированный порядок загрузки с уровнями"""
        if self.should_skip_package(start_package):
            return []

        visited = set()
        load_order = []
        level = 0

        def dfs_detailed(node: str, current_level: int):
            if self.should_skip_package(node) or node in visited:
                return
            visited.add(node)

            # Добавляем узел с информацией об уровне
            load_order.append({
                'package': node,
                'level': current_level
            })

            # Рекурсивно обходим зависимости с увеличенным уровнем
            for neighbor in self.graph.get(node, []):
                if not self.should_skip_package(neighbor):
                    dfs_detailed(neighbor, current_level + 1)

        dfs_detailed(start_package, level)
        return load_order

    def generate_graphviz_dot(self, start_package: str) -> str: # Формирование текстового представления графа зависимостей на языке диаграмм Graphviz
        filtered_graph = self.get_filtered_graph()

        dot_lines = [
            "digraph Dependencies {",
            "  rankdir=TB;",
            "  node [shape=box, style=filled, fillcolor=lightblue];",
            f'  "{start_package}" [fillcolor=lightcoral, shape=ellipse];'
        ]

        # Добавляем узлы и ребра
        for package, deps in filtered_graph.items():
            if deps:
                for dep in deps:
                    # Определяем цвет ребра для циклов
                    edge_color = "red" if self._is_edge_in_cycle(package, dep) else "black"
                    dot_lines.append(f'  "{package}" -> "{dep}" [color="{edge_color}"];')
            else:
                # Листовые узлы
                if package != start_package:
                    dot_lines.append(f'  "{package}" [fillcolor=lightgreen];')

        dot_lines.append("}")
        return "\n".join(dot_lines)

    def _is_edge_in_cycle(self, source: str, target: str) -> bool:
        """Проверяет, является ли ребро частью цикла"""
        for cycle in self.cycles:
            for i in range(len(cycle) - 1):
                if cycle[i] == source and cycle[i + 1] == target:
                    return True
            # Проверяем замыкание цикла
            if cycle[-1] == source and cycle[0] == target:
                return True
        return False


class PackageManagerSimulator:
    """Симулятор менеджера пакетов для сравнения результатов"""

    @staticmethod
    def simulate_npm_install_order(dependencies: Dict[str, List[str]], package: str) -> List[str]:

        visited = set()
        install_order = []

        def dfs_npm(node: str):
            if node in visited:
                return
            visited.add(node)

            # Сортируем зависимости в алфавитном порядке как npm
            sorted_deps = sorted(dependencies.get(node, []))
            for dep in sorted_deps:
                dfs_npm(dep)

            install_order.append(node)

        dfs_npm(package)
        return install_order

    @staticmethod
    def simulate_pip_install_order(dependencies: Dict[str, List[str]], package: str) -> List[str]:

        visited = set()
        install_order = []

        def dfs_pip(node: str):
            if node in visited:
                return
            visited.add(node)

            # Оригинальный порядок зависимостей как в requirements
            for dep in dependencies.get(node, []):
                dfs_pip(dep)

            install_order.append(node)

        dfs_pip(package)
        return install_order


class DependencyVisualizer:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.url_parser = URLRepositoryParser()
        self.npm_parser = NPMPackageParser()
        self.test_parser = TestRepositoryParser()
        self.dependency_graph = DependencyGraph()
        self.package_manager_simulator = PackageManagerSimulator()

    def is_url(self, repository: str) -> bool:
        return repository.startswith(('http://', 'https://'))

    def validate_parameters(self) -> bool: # обработка ошибок

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

    def display_config(self): # параметры, настраиваемые пользователем
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

            # Если включен режим порядка загрузки
            if self.config.get('show_load_order', False):
                self.analyze_load_order()

            # Визуализируем граф
            if not self.config.get('no_visualization', False):
                self.visualize_dependency_graph()

            print(f"\nРезультат сохранен в: {self.config['output_file']}")

        except ValueError as e:
            print(f"Ошибка анализа: {e}")
            return False

        return True

    def build_dependency_graph(self, dependencies_data: Dict[str, Any]):
        if self.config['test_mode']:
            # Для тестовых данных (уже готовый граф)
            print("Построение графа из тестовых данных...")
            for package, deps in dependencies_data.items():
                for dep in deps:
                    if dep:  # проверяем, что зависимость не пустая
                        self.dependency_graph.add_dependency(package, dep)
        else:
            print("Построение графа из npm зависимостей...")
            for package, version in dependencies_data.items():
                self.dependency_graph.add_dependency(self.config['package_name'], package)

    def analyze_dependency_graph(self):
        # Выводим структуру графа для отладки
        self.dependency_graph.print_graph()

        print("АНАЛИЗ ГРАФА ЗАВИСИМОСТЕЙ")

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

    def analyze_load_order(self):
        """Анализирует и сравнивает порядок загрузки зависимостей"""
        print("АНАЛИЗ ПОРЯДКА ЗАГРУЗКИ ЗАВИСИМОСТЕЙ")

        # Порядок загрузки нашим алгоритмом
        load_order = self.dependency_graph.get_load_order(self.config['package_name'])
        detailed_order = self.dependency_graph.get_detailed_load_order(self.config['package_name'])

        print(f"\nПорядок загрузки для '{self.config['package_name']}':")
        print("(глубина -> пакет)")
        for item in detailed_order:
            level_indent = "  " * item['level']
            print(f"{level_indent}L{item['level']} -> {item['package']}")

        print(f"\nЛинейный порядок загрузки: {' -> '.join(load_order)}")

        # Сравнение с менеджерами пакетов
        if self.config['test_mode']:
            self.compare_with_package_managers()

    def compare_with_package_managers(self):
        print("СРАВНЕНИЕ С МЕНЕДЖЕРАМИ ПАКЕТОВ")

        # Наш порядок
        our_order = self.dependency_graph.get_load_order(self.config['package_name'])

        # Симуляция npm
        npm_order = self.package_manager_simulator.simulate_npm_install_order(
            dict(self.dependency_graph.graph),
            self.config['package_name']
        )

        # Симуляция pip
        pip_order = self.package_manager_simulator.simulate_pip_install_order(
            dict(self.dependency_graph.graph),
            self.config['package_name']
        )

        print(f"\nНаш алгоритм:    {' -> '.join(our_order)}")
        print(f"NPM-подобный:    {' -> '.join(npm_order)}")
        print(f"PIP-подобный:    {' -> '.join(pip_order)}")

        # Анализ расхождений
        differences = []
        if our_order != npm_order:
            differences.append("NPM сортирует зависимости алфавитно на каждом уровне")
        if our_order != pip_order:
            differences.append("PIP сохраняет оригинальный порядок зависимостей")
        if our_order == npm_order == pip_order:
            differences.append("Все алгоритмы дают одинаковый результат")

        print(f"\nРасхождения:")
        for diff in differences:
            print(f"  - {diff}")

    def visualize_dependency_graph(self): # изображение графа в файле формата SVG
        try:
            # Генерируем DOT представление
            dot_source = self.dependency_graph.generate_graphviz_dot(self.config['package_name'])

            # Сохраняем DOT файл
            dot_filename = self.config['output_file'].replace('.svg', '.dot')
            with open(dot_filename, 'w', encoding='utf-8') as f:
                f.write(dot_source)
            print(f"Текстовое представление Graphviz сохранено в: {dot_filename}")

            # Создаем граф из DOT источника
            dot = graphviz.Source(dot_source)

            # Сохраняем в SVG
            svg_filename = self.config['output_file'].replace('.png', '.svg')
            dot.render(filename=svg_filename.replace('.svg', ''), format='svg', cleanup=True)

            print(f"Визуализация графа сохранена в: {svg_filename}")

        except Exception as e:
            print(f"Ошибка при визуализации графа: {e}")
            import traceback
            traceback.print_exc()

    def filter_dependencies(self, dependencies: Dict[str, str]) -> Dict[str, str]:
        if not self.config['filter_substring']:
            return dependencies

        filtered = {}
        for package, version in dependencies.items():
            if self.config['filter_substring'].lower() in package.lower():
                filtered[package] = version

        return filtered

    def display_direct_dependencies(self, dependencies: Dict[str, str]): # все прямые зависимости заданного пользователем пакета
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
    print("ДЕМОНСТРАЦИЯ ФУНКЦИОНАЛЬНОСТИ")

    # Создаем тестовые файлы
    create_test_files()

    test_cases = [
        {
            "name": "Линейный граф",
            "package": "A",
            "repo": "test_linear.txt",
            "filter": "",
            "test_mode": True,
            "show_load_order": True
        },
        {
            "name": "Ветвящийся граф с фильтром 'D'",
            "package": "A",
            "repo": "test_branching.txt",
            "filter": "D",
            "test_mode": True,
            "show_load_order": True
        },
        {
            "name": "Циклический граф",
            "package": "A",
            "repo": "test_cyclic.json",
            "filter": "",
            "test_mode": True,
            "show_load_order": True
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'─' * 50}")
        print(f"ТЕСТ {i}: {test_case['name']}")
        print(f"{'─' * 50}")

        config = {
            'package_name': test_case['package'],
            'repository': test_case['repo'],
            'output_file': f'test_output_{i}.svg',
            'test_mode': test_case['test_mode'],
            'filter_substring': test_case['filter'],
            'show_load_order': test_case['show_load_order']
        }

        visualizer = DependencyVisualizer(config)
        visualizer.analyze_dependencies()


def parse_arguments(): #параметры командной строки
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
        default='dependency_graph.svg',
        help='Имя сгенерированного файла с изображением графа (SVG)'
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

    parser.add_argument(
        '--load-order',
        dest='show_load_order',
        action='store_true',
        default=False,
        help='Показать порядок загрузки зависимостей и сравнить с менеджерами пакетов'
    )

    parser.add_argument(
        '--no-viz',
        dest='no_visualization',
        action='store_true',
        default=False,
        help='Не создавать визуализацию графа'
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
            'filter_substring': args.filter_substring,
            'show_load_order': args.show_load_order,
            'no_visualization': args.no_visualization
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

"""
Тесты

# Демонстрация всех функций
python main.py --demo

# Отдельные тесты
python main.py --package A --repo test_linear.txt --test-mode --output linear.svg
python main.py --package A --repo test_branching.txt --test-mode --filter D --output branching.svg
python main.py --package A --repo test_cyclic.json --test-mode --output cyclic.svg

# Пакет с production зависимостями (Express.js)
python main.py --package express --repo https://github.com/expressjs/express --output express_deps.svg

# Пакет с mixed зависимостями (Vue.js)
python main.py --package vue --repo https://github.com/vuejs/vue --output vue_mixed.svg

# Пакет с TypeScript (много dev зависимостей)
python main.py --package typescript --repo https://github.com/microsoft/TypeScript --output typescript_deps.svg
"""