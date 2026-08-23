[English](README.md) | **Русский**

# SkyrimNet Chatterbox TTS Server

Самодостаточный FastAPI-сервер, который позволяет SKSE-плагину
[SkyrimNet](https://github.com/MinLL/SkyrimNet-GamePlugin) использовать
[Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox)
как локальный TTS-движок с **клонированием голоса** и **контролем
эмоций** — всё на одной GPU на 8 ГБ.

Chatterbox нет в списке движков, поставляемых с SkyrimNet
(Piper / PocketTTS / XTTS / ElevenLabs / Inworld / Zonos), а generic'овые
Chatterbox-UI на Hugging Face не говорят по проводному протоколу
`GradioTTSInterface`, который использует C++-клиент SkyrimNet.
Этот проект связывает их между собой.

## Возможности

- Локальный инференс. Никаких облачных API и лимитов запросов.
- 23 языка из коробки: русский, английский, немецкий и т.д.
- Живое клонирование голоса из любого референсного WAV, который
  загружает SkyrimNet.
- Эмоциональные теги из диалогов SkyrimNet (`[shout]`, `[whisper]`,
  `[angry]`, `[sad]`, `[happy]`, `[dramatic]`, ...) мапятся на
  значения (exaggeration, cfg_weight) для каждой эмоции, чтобы они
  реально меняли подачу речи — Chatterbox чисто акустическая модель
  и иначе произнесёт тег буквально как слово.
- Один файл, без фреймворка Gradio, без Node.js, без WSL.

## Требования

- Python 3.10+
- NVIDIA GPU с ~4 ГБ VRAM и CUDA 11.8 или
  12.x. Режим CPU тоже работает, но 30 секунд диалога на CPU
  займут около 3 минут.
- [SkyrimNet](https://github.com/MinLL/SkyrimNet-GamePlugin),
  установленный в папку `SKSE/Plugins/` вашего Skyrim Special Edition
  (этот проект даёт только сервер; сам мод распространяется отдельно).

## Установка

```bash
git clone https://github.com/DigitalJesus/skyrimnet-chatterbox-server.git
cd skyrimnet-chatterbox-server
pip install -r requirements.txt
```

При первом запуске веса мультиязычной модели (~2 ГБ) скачаются
с Hugging Face в локальный кэш (`~/.cache/huggingface/`).

## Запуск

```bash
python server.py
# или на Windows:
start.bat
```

По умолчанию сервер слушает `http://127.0.0.1:7861`. Откройте
`http://127.0.0.1:7861/health`, чтобы убедиться, что модель загрузилась
(`"model_loaded": true`).

В настройках TTS SkyrimNet укажите Gradio server URL:
`http://127.0.0.1:7861` — и выберите любой голосовой профиль.
Сэмплы голоса мод загружает сам и кэширует их в
`%LOCALAPPDATA%\Temp\gradio` (или туда, куда указывает
`GRADIO_UPLOAD_DIR`).

## Настройка

Все параметры задаются переменными окружения с разумными дефолтами
для Windows + Skyrim.

| Переменная              | Дефолт                                                  | Что делает |
|-------------------------|---------------------------------------------------------|------------|
| `GRADIO_HOST`           | `127.0.0.1`                                             | Адрес привязки |
| `GRADIO_PORT`           | `7861`                                                  | Порт привязки |
| `GRADIO_UPLOAD_DIR`     | `%LOCALAPPDATA%\Temp\gradio`                            | Где кэшируются сэмплы голоса и сгенерированные WAV |
| `DEFAULT_VOICE_SAMPLE`  | `voice-sample.mp3` рядом со скриптом                    | Резервный голос, если SkyrimNet не прислал референс |
| `MODEL_REPO`            | `ResembleAI/chatterbox`                                 | Репозиторий Hugging Face для загрузки |
| `DEVICE`                | `cuda`, если доступен, иначе `cpu`                      | Torch-устройство |

## Эмоциональные теги

SkyrimNet добавляет к тексту диалога теги в скобках, например:

> «Беги, дурак! [shout]»

Сервер распознаёт следующие теги и подставляет соответствующий
пресет Chatterbox:

| Тег                       | exaggeration | cfg_weight | Ощущение |
|---------------------------|--------------|------------|----------|
| `[whisper]` `[whispering]`| 0.25         | 0.70       | тихо, осторожно |
| `[quiet]`                 | 0.30         | 0.65       | мягко |
| `[sad]` `[somber]`        | 0.40         | 0.70       | низкая энергия |
| `[fear]` `[scared]`       | 0.70         | 0.40       | нервозность |
| `[terrified]`             | 0.85         | 0.35       | паника |
| `[angry]` `[dramatic]`    | 0.75         | 0.35       | интенсивно, театрально |
| `[shout]` `[yelling]`     | 0.85         | 0.30       | громко |
| `[scream]`                | 0.95         | 0.25       | крик |
| `[happy]` `[cheerful]`    | 0.60         | 0.45       | бодро |
| `[laugh]` `[laughing]`    | 0.65         | 0.40       | смех |
| `[sarcastic]`             | 0.55         | 0.55       | сухо |
| `[neutral]`               | 0.50         | 0.50       | ровно |
| (без тега)                | 0.35         | 0.35       | дефолт SkyrimNet |

Если вы подняли `exaggeration` и `cfg_weight` выше 0.45 в UI
SkyrimNet, ваши значения сохраняются, а эмоция тега игнорируется
(глобальная настройка важнее).

## Протокол

Сервер реализует то малое подмножество queue-API Gradio, которое
реально вызывает `GradioTTSInterface` из SkyrimNet. Три эндпоинта
плюс один файловый роут:

```
POST  /gradio_api/upload                multipart      -> ["abs/path.wav"]
POST  /gradio_api/call/generate_audio   {data:[...]}  -> {"event_id": "..."}
GET   /gradio_api/call/generate_audio/{eid}            -> SSE: "event: complete\ndata: {...}\n\n"
GET   /gradio_api/file=<abs_path>                     -> audio/wav bytes
```

Массив `data` из 30 элементов парсится по типам, а не по позициям,
потому что порядок аргументов, в котором SkyrimNet их шлёт, не
совпадает ни с одним из двух эталонных файлов `gradio_tts_app.py`,
опубликованных Resemble AI. Парсер берёт самую длинную строку (текст),
двухбуквенный код языка, путь до wav, который не является заглушкой
`empty_100ms.wav` (референс голоса), и три числа с плавающей точкой,
попадающие в диапазоны `(exaggeration, cfg_weight, temperature)`
соответственно.

Poll-эндпоинт возвращает однострочный SSE-ответ (не
`StreamingResponse` с chunked transfer), потому что некоторые
Gradio-клиентские библиотеки парсят последнее как преждевременно
закрывшийся поток. Байты WAV инлайнятся в base64 в полях
`bytes` / `bytes_b64` / `data`, так что клиенту SkyrimNet не нужен
повторный GET на `/file=...`.

## Решение проблем

- **`Model load failed: ChatterboxMultilingualTTS.from_pretrained() got an unexpected keyword argument 'cache_dir'`** —
  это от старого форка. Поставьте официальный пакет:
  `pip install -U chatterbox-tts`.

- **Длинный звук обрывается на середине фразы с `forcing EOS token`** —
  у Chatterbox есть внутренний детектор повторов, который принудительно
  завершает генерацию, когда декодер зацикливается. Попробуйте понизить
  `cfg_weight` (например до 0.25) или упростить текст.

- **SkyrimNet всё ещё показывает `Failed to process TTS response`** — убедитесь,
  что URL сервера в настройках TTS мода — ровно
  `http://127.0.0.1:7861` (без слэша на конце, не `localhost`). SSE-ответ
  с поллингом несёт аудио прямо внутри себя в base64, так что отдельный
  запрос `GET /file=...` не нужен, но SkyrimNet сначала делает `HEAD`-запрос
  к URL файла; этот эндпоинт должен вернуть 200.

- **У вас 8 ГБ VRAM и модель на 3 ГБ** — всё нормально. Chatterbox
  Multilingual — 500 млн параметров; на 4060 грузится секунд за 30
  и синтезирует строку длиной 3 секунды примерно за 1 секунду.

## Лицензия

MIT. См. `LICENSE`. Проект не включает и не распространяет SkyrimNet,
Skyrim или веса модели Chatterbox; пользователи должны получить каждое
из них из соответствующих источников на условиях их собственных лицензий.
