# MovieMatch

MovieMatch — рекомендательная система фильмов с машинным обучением.
На текущем этапе доступны FastAPI API, стартовая страница Streamlit,
загрузка и проверка CSV MovieLens, исследовательский анализ данных (EDA).
ML-модели и авторизация пока отсутствуют.

## Технологии

Python 3.14, FastAPI, Streamlit, Pandas, NumPy, Matplotlib, Scikit-learn, SQLite и Pytest.
SQLite доступен через стандартный модуль `sqlite3`; отдельная установка не нужна.
`httpx` используется клиентом API в тестах, `uvicorn` — для запуска FastAPI.

## Структура

```text
backend/       # FastAPI и маршрут /health
frontend/      # Интерфейс Streamlit
ml/            # data_loader.py: проверка CSV; eda.py: анализ и графики
data/          # Локальная папка ml-latest-small с настоящим MovieLens
reports/       # Автоматические summary.json и figures/*.png
notebooks/     # Будущие исследовательские ноутбуки
tests/         # Автоматические тесты
requirements.txt
```

## Установка и запуск на Windows

Все команды выполняются в PowerShell из корневой директории MovieMatch.
Используется уже существующее окружение `venv`; создавать новое не нужно.
Явный путь к Python позволяет работать без активации окружения и изменения
политики выполнения PowerShell.

Проверьте Python (ожидается версия 3.14) и установите зависимости:

```powershell
.\venv\Scripts\python.exe --version
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m pip check
```

Запустите API:

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Проверка: <http://127.0.0.1:8000/health> возвращает `{"status":"ok"}`.
Документация API: <http://127.0.0.1:8000/docs>.

Во втором терминале из корня проекта запустите интерфейс:

```powershell
.\venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Откройте <http://localhost:8501>. Для остановки каждого сервера нажмите `Ctrl+C`.
На этом этапе интерфейс показывает сообщение о разработке и не обращается к API.

## Тесты

```powershell
.\venv\Scripts\python.exe -m pytest
```

Тесты проверяют `/health`, загрузчик, ошибки данных, статистику и создание графиков.
Проверки данных используют небольшие искусственные CSV во временных каталогах;
скачивание MovieLens и отдельный запуск API для тестов не нужны.

## Данные MovieLens Latest Small

Источник: [GroupLens MovieLens Latest Small](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip).
Описание формата и условия использования находятся в
[README GroupLens](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html).
Версия Latest может обновляться; результаты анализа вычисляются по локальным CSV.

Из корня проекта в PowerShell скачайте и распакуйте архив:

```powershell
Invoke-WebRequest -Uri "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip" -OutFile "data/ml-latest-small.zip"
Expand-Archive -LiteralPath "data/ml-latest-small.zip" -DestinationPath "data" -Force
```

Если сеть недоступна, скачайте архив по ссылке в браузере на компьютере с доступом
к сети, перенесите его и распакуйте в `data/`. Ожидаются файлы:

```text
data/ml-latest-small/movies.csv
data/ml-latest-small/ratings.csv
data/ml-latest-small/tags.csv
data/ml-latest-small/links.csv
```

Фиктивный датасет для рабочего анализа не создаётся. CSV не входят в Git.
Загрузчик `ml.data_loader.load_data()` читает UTF-8 CSV и возвращает словарь
DataFrame с ключами `movies`, `ratings`, `tags`, `links`. Все четыре файла и
обязательные столбцы проверяются до анализа. ID и timestamp приводятся к nullable
`Int64`, rating — к `Float64`, текст — к `string`. ID должны быть положительными,
timestamp — неотрицательным целым числом секунд Unix, оценки — от 0.5 до 5 с шагом 0.5.
Непустые значения неверного типа вызывают `ValueError` с именем файла и столбца;
отсутствующие файлы — `FileNotFoundError`. Дополнительные столбцы сохраняются.

Пропуски и дубликаты сохраняются для исследования, не удаляются и не заполняются.
Пустые поля и текстовые поля из пробелов считаются пропусками; строка `NA`
сохраняется как текст. Внешние ссылки IMDb/TMDb могут отсутствовать.

## Запуск EDA

Обновите зависимости и запустите анализ из корня проекта:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m ml.eda
```

Можно указать другие директории:

```powershell
.\venv\Scripts\python.exe -m ml.eda --data-dir data/ml-latest-small --output-dir reports
```

Результаты (повторный запуск перезаписывает файлы):

- `reports/summary.json`: количества фильмов, пользователей, оценок и тегов;
  пропуски по столбцам, полные дубликаты строк и дубликаты ключей; общий средний
  рейтинг, средние по пользователям и фильмам, частоты оценок и жанров, топ фильмов.
- `reports/figures/rating_distribution.png`: распределение оценок.
- `reports/figures/ratings_per_movie.png`: распределение числа оценок на фильм,
  включая фильмы без оценок.
- `reports/figures/top_10_genres.png`: десять жанров с наибольшим количеством фильмов.
- `reports/figures/top_10_movies.png`: десять фильмов с наибольшим количеством оценок.

Пользователи считаются по объединению ratings и tags; отдельно выводится число
пользователей с оценками. Средние и число оценок фильма не учитывают пустые rating;
общее число строк ratings включает их. Повторные оценки сохраняют свой вес.
Жанры разделяются по `|`; один фильм может учитываться в нескольких жанрах,
маркер `(no genres listed)` не считается жанром. При повторном movieId метаданные
берутся из первой строки, неизвестные movieId из ratings остаются в статистике.
Графики сохраняются через Matplotlib Agg: окно GUI и запущенные серверы не нужны.
Автоматические отчёты исключены из Git; обучение модели не выполняется.

## Совместимость

В `requirements.txt` заданы диапазоны версий с поддержкой Python 3.14.
Для научных библиотек на Windows используйте 64-битный CPython.
Если pip сообщает об отсутствии подходящей сборки, проверьте версию и архитектуру
Python и обновите pip в существующем окружении:

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
```

Локальные окружения, `.env`, кэш Python и файлы SQLite исключены через `.gitignore`.
