# Датасеты: размер, содержание, скачивание

## Быстрый старт (маленький набор)

```bash
cd tools/train
pip install -r requirements.txt
python download_samples.py --out ./raw/samples --count 100
python augment.py --input ./raw/samples --output ./dataset --count 5000
python train.py --data ./dataset --epochs 15 --out ./checkpoints/model.pt
```

Очистка синтетики (`images/` + `labels.json`). Чекпоинты и веса **не** удаляются:

```bash
python clean_dataset.py              # только ./dataset
python clean_dataset.py --all -y     # dataset + raw, модели остаются
```

| Что | Значение |
|-----|----------|
| Источник | [picsum.photos](https://picsum.photos) (случайные сток-фото) |
| Объём | 50–200 JPG, ~512×512 |
| На диске | ~10–40 МБ |
| Назначение | проверить пайплайн end-to-end, **не** для сдачи качества |

## Основной набор — DIV2K

```bash
python download_div2k.py --out ./raw/div2k          # valid, все PNG из zip
python download_div2k.py --out ./raw/div2k --limit 100   # скачать zip, распаковать только 100
# python download_div2k.py --out ./raw/div2k --split train   # train, ~3.7 GB zip
python augment.py --input ./raw/div2k/DIV2K_valid_HR --output ./dataset --count 20000
```

> Zip DIV2K всё равно качается целиком (~450 МБ valid). `--limit` ограничивает только число **распакованных** PNG.

| Сплит | Картинок | Zip | После распаковки | Содержание |
|-------|----------|-----|------------------|------------|
| **valid** (по умолчанию) | ~100 HR PNG | ~450 МБ | ~0.5–1 ГБ | Разнообразные сцены высокого разрешения (пейзажи, город, интерьеры) |
| **train** | ~800 HR PNG | ~3.7 ГБ | ~4–8 ГБ | То же, больше объём |

DIV2K — классика для супер-разрешения; кадры уже «хорошие», идеальны для **синтетической порчи** (яркость/контраст/насыщенность).

Лицензия: для исследований; для коммерческой сдачи уточните условия ETHZ / используйте свои фото.

## Сколько примеров для TinyCNN

Речь о **синтетических сэмплах после `augment.py`**, не о числе исходных фото.

| Цель | Исходных фото | Синтетики (`--count`) | Эпохи |
|------|---------------|------------------------|-------|
| Smoke-test | 50–100 | 1k–2k | 5–10 |
| Рабочий baseline | 100–800 (DIV2K valid) | **10k–20k** | 15–30 |
| Лучше обобщение | + свои/COCO кусок | **20k–50k** | 30–50 |

Уникальных исходников важнее «тупо больше копий одной картинки»: лучше 800 разных × аугментации, чем 20 фото × 50k одинаковых сцен.

Что лежит в `./dataset` после augment:

```
dataset/
  images/000000.jpg …   # bad = inverse_apply(good, P)
  labels.json           # P = brightness, contrast, saturation (+ recon_mse)
  meta.json             # формула apply-consistent-v1
```

Старый датасет (Pillow-labels) лучше пересобрать: `python clean_dataset.py -y` → снова `augment.py`.

## Лица (аватарки) и аниме

```bash
python download_domain.py --preset faces --out ./raw/faces --limit 300
python download_domain.py --preset anime --out ./raw/anime --limit 300
# при необходимости: --hf-id other/dataset-on-huggingface
```

| Preset | Источник | Содержание | Размер порядка |
|--------|----------|------------|----------------|
| **faces** | [LFW](http://vis-www.cs.umass.edu/lfw/) | реальные лица / «аватарки» | архив ~170 МБ, потом `--limit` файлов |
| **anime** | HuggingFace (`cchen856/anime-faces` по умолчанию) | аниме-лица | зависит от `--limit`; нужен `datasets` |

Смешать с DIV2K (рекомендуется для обобщения):

```bash
mkdir raw\mixed
xcopy /E /I raw\div2k\DIV2K_valid_HR raw\mixed\div2k
xcopy /E /I raw\faces raw\mixed\faces
xcopy /E /I raw\anime raw\mixed\anime
python augment.py --input ./raw/mixed --output ./dataset --count 20000
```

> Разнообразие доменов (фото / лица / аниме) сильнее влияет на устойчивость, чем ещё 10k копий одного DIV2K.  
> Аниме-датасеты часто имеют NSFW и разные лицензии — проверяйте перед сдачей.

## Другие источники (вручную / позже)

| Датасет | Зачем | Размер порядка |
|---------|-------|----------------|
| COCO / Open Images | быт, объекты, улица | десятки ГБ |
| FFHQ / CelebA-HQ | высококачественные лица | несколько ГБ |
| MIT-Adobe FiveK | пары «сырое → ретушь» | ~50 ГБ |
| PPR10K | портретная ретушь | несколько ГБ |
| Danbooru (полный) | максимальное аниме-разнообразие | очень большой |
| Своя съёмка / свои аватары | контроль прав и сцен | по желанию |

Автоскачивание в репо: **samples**, **DIV2K**, **faces (LFW)**, **anime (HF)**.

## Диск под полный цикл (ориентир)

| Этап | Место |
|------|-------|
| samples | &lt; 50 МБ |
| DIV2K valid + zip | ~1 ГБ |
| dataset 20k JPG 224 | ~0.5–1.5 ГБ |
| checkpoint .pt + onnx | &lt; 50 МБ |
