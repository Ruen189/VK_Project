# VK Image Enhancer

Клиентский модуль улучшения изображений **в браузере**: нейросеть (или эвристика) подбирает параметры яркости / контраста / цветности, затем `apply` применяет их к полному кадру в Web Worker (UI не блокируется).

| Требование | Статус |
|------------|--------|
| Современные браузеры | Chrome / Firefox / Edge / Safari |
| Форматы | JPG, PNG, BMP, HEIC |
| До 15 Мпк, ≤30 с (цель avg ~5 с) | JS-apply; на больших кадрах может быть медленнее цели |
| Бандл ≤10 МБ | `npm run check:size` (~1.4 МБ с HEIC-decoder) |
| Task API + progress / cancel | да |

## Быстрый старт

```bash
npm install
npm test
npm run demo
```

Откроется **http://127.0.0.1:5173/** (если нет — откройте вручную). Остановка: `Ctrl+C`.

В demo: выбор файла → опционально «Синтетически испортить» → «Улучшить».  
Предиктор: **эвристика** или **TinyCNN** (нужен `models/enhance_params.json`).

## API

```ts
import { ImageEnhancer } from './dist/index.js'; // или ваш путь к пакету

const enhancer = new ImageEnhancer({
  workerUrl: new URL('./dist/worker.js', import.meta.url),
  heicDecoderUrl: new URL('./dist/vendor/heic2any.js', import.meta.url),
});

enhancer.on('status', (info) => {
  console.log(info.status, info.progress, info.metrics?.params);
});

const id = await enhancer.submit(file, {
  outputType: 'image/jpeg',
  predictorMode: 'heuristic', // или 'model'
  modelUrl: './models/enhance_params.json', // для mode: 'model'
});

const blob = await enhancer.getResult(id);
enhancer.cancel(id);
enhancer.dispose();
```

| Метод / событие | Назначение |
|-----------------|------------|
| `submit` | поставить задачу, вернуть `taskId` |
| `getStatus` | статус и progress |
| `cancel` | прервать задачу |
| `getResult` | готовый `Blob` |
| `on('status')` | события смены статуса / прогресса |

`b` / `c` / `s` в UI — предсказанные **brightness**, **contrast**, **saturation**.

## Структура

```
src/            API, Worker, decode/apply, эвристика, TinyCNN
demo/           страница для ручной проверки
models/         enhance_params.json после экспорта весов
tools/train/    датасеты, augment, PyTorch train/export
scripts/        build helpers (HEIC bundle, demo server, size check)
benchmarks/     место под эталонные кадры приёмки
```

## Обучение модели

1. Скачать исходники → 2. `augment.py` (labels в пространстве `apply`) → 3. `train.py` → 4. `export_web_weights.py` → 5. `npm run demo`, режим «Модель».

```bash
cd tools/train
pip install -r requirements.txt

# данные (любой микс)
python download_samples.py --out ./raw/samples --count 100
# python download_div2k.py --out ./raw/div2k
# python download_domain.py --preset faces --out ./raw/faces --limit 300
# python download_domain.py --preset anime --out ./raw/anime --limit 300

python clean_dataset.py -y          # только ./dataset, не raw и не checkpoints
python augment.py --input ./raw/div2k/DIV2K_valid_HR --output ./dataset --count 10000
python train.py --data ./dataset --epochs 20 --out ./checkpoints/model.pt
python export_web_weights.py --checkpoint ./checkpoints/model.pt --out ../../models/enhance_params.json
```

Повторный `train` без `--overwrite` пишет `model_2.pt`, `model_3.pt`, …  
После смены JSON: `npm run build` и жёсткое обновление страницы demo.

Подробности по датасетам: `tools/train/DATASETS.md`.

## Чего ещё не хватает до идеала

- **Замеры** тестирование на Safari затруднено
- **Очередь задач** — сейчас по сути одна активная обработка в Worker
- **Прод-сборка npm-пакета** (публичный entry, semver) при необходимости платформы
- **Разнообразный датасет** под сдачу (микс photo / faces / anime)
- Опционально: INT8 / ONNX Runtime Web вместо JSON-весов; lazy-HEIC отдельным chunk ещё сильнее ужать «core»
