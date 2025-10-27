import argparse
import sys
import os
import json
import requests
from typing import Dict, Any


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


class DependencyVisualizer:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.url_parser = URLRepositoryParser()
        self.npm_parser = NPMPackageParser()

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
            if self.is_url(self.config['repository']):
                print("Режим: URL репозитория")
                dependencies = self.url_parser.parse_from_url(
                    self.config['repository'],
                    self.config['package_name']
                )
            else:
                print("Режим: Локальный файл")
                dependencies = self.npm_parser.get_dependencies(self.config['package_name'])

            if self.config['filter_substring']:
                dependencies = self.filter_dependencies(dependencies)
                print(f"Применен фильтр: '{self.config['filter_substring']}'")

            self.display_direct_dependencies(dependencies)

            print(f"\nРезультат будет сохранен в: {self.config['output_file']}")

        except ValueError as e:
            print(f"Ошибка анализа: {e}")
            return False

        return True

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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Инструмент визуализации графа зависимостей пакетов',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '--package', '--package-name',
        dest='package_name',
        required=True,
        help='Имя анализируемого npm пакета'
    )

    parser.add_argument(
        '--repo', '--repository',
        dest='repository',
        required=True,
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