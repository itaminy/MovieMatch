# MovieMatch

MovieMatch — рекомендательная система фильмов с машинным обучением.
На текущем этапе создан каркас: FastAPI API, стартовая страница Streamlit
и тест доступности API. ML-модели, датасеты и авторизация пока отсутствуют.

## Технологии

Python 3.14, FastAPI, Streamlit, Pandas, NumPy, Scikit-learn, SQLite и Pytest.
SQLite доступен через стандартный модуль `sqlite3`; отдельная установка не нужна.
`httpx` используется клиентом API в тестах, `uvicorn` — для запуска FastAPI.

## Структура

```text
backend/       # FastAPI и маршрут /health
frontend/      # Интерфейс Streamlit
ml/            # Будущие компоненты машинного обучения
data/          # Будущие данные
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

Тест проверяет HTTP-статус и JSON маршрута `/health`; отдельный запуск API не нужен.

## Совместимость

В `requirements.txt` заданы диапазоны версий с поддержкой Python 3.14.
Для научных библиотек на Windows используйте 64-битный CPython.
Если pip сообщает об отсутствии подходящей сборки, проверьте версию и архитектуру
Python и обновите pip в существующем окружении:

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
```

Локальные окружения, `.env`, кэш Python и файлы SQLite исключены через `.gitignore`.
