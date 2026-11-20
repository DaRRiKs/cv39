# №5. MNIST + LeNet-5, CPU

Код: `train.py`, `evaluate.py`, `cvkit.py`. Отчёт: [report.md](report.md). Результаты: `results/`.

**Сейчас работа не завершена:** среда и код проверены, но torchvision не смог скачать MNIST/KMNIST (`WinError 10061`). Полные обучения, реальный MNIST resume, метрики и изображения по данным не получены. Причины — `results/data_availability.json`; невыполненные критерии — `results/checks/criteria.csv`.

## Установка и данные

Команды запускаются из корня проекта. Проверено на Windows, Python 3.14.5, CPU float32. Версии закреплены в `requirements.txt`, полный список — `results/pip-freeze.txt`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
python evaluate.py --self-check
python evaluate.py --prepare
```

Если `.venv` уже существует, повторно создавать его не нужно. Для Linux/macOS: `source .venv/bin/activate`.

`--prepare` сохраняет EDA, лист обработки, стратифицированные индексы train/val 54 000/6 000 и статистики **только MNIST train**. Seed=1005, два потока PyTorch. Официальный test оценивается в конце. После восстановления сети повторите `--prepare`.

Для работы без сети можно перенести официальные распакованные файлы `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte` в `data/MNIST/raw/` и `data/KMNIST/raw/`, каждый домен в свою папку. Источники: [MNIST](https://docs.pytorch.org/vision/0.24/generated/torchvision.datasets.MNIST.html), [KMNIST](https://github.com/rois-codh/kmnist).

## Baseline, обучения и resume

```powershell
python evaluate.py --classical all
python train.py --dataset mnist --aug none --out_dir results/none
python train.py --dataset mnist --aug rotate10 --out_dir results/rotate10 --stop-after 7
python train.py --dataset mnist --aug rotate10 --out_dir results/rotate10 --resume results/rotate10/last.pt
```

Два отдельных StandardScaler→kNN(k=5, Euclidean) используют все train-изображения. Вместо `all` доступны `raw` и `hog`.

Оба LeNet-прогона имеют одинаковые начальные веса, SGD lr=0.1, momentum=0.9, weight_decay=5e-4, cosine, batch=128 и **ровно 15 эпох**, early stopping выключен. Rotate10 останавливается после эпохи 7; отдельный процесс продолжает эпохи 8–15 с сохранёнными optimizer, scheduler и всеми RNG. CSV не дублируется. Подтверждение — `results/rotate10/events.jsonl`.

## Val, ошибки и интерпретация

```powershell
python evaluate.py --ckpt results/none/best.pt --dataset mnist
python evaluate.py --ckpt results/rotate10/best.pt --dataset mnist --errors --gradcam --robustness --latency
python evaluate.py --plots
python evaluate.py --review-templates
```

Результаты включают общие и поклассовые метрики, confusion matrix, CSV и галереи **всех** ошибок, топ-5 пар, Grad-CAM 5+5, robustness и CPU latency 20+100. Кривые строятся по CSV.

После просмотра реальных галерей заполните в `results/analysis/error_review.csv` поля `group`, `observation`, `hypothesis` для всех ошибок; минимум 10 подробно разобранных случаев отметьте `detailed=1` (если ошибок меньше — все). В `gradcam_review.csv` заполните наблюдение и гипотезу для каждой карты. До получения изображений этот анализ выполнить нельзя. Повторная генерация шаблонов не стирает записи.

## MNIST → KMNIST

```powershell
python train.py --dataset kmnist --aug rotate10 --epochs 10 --lr 0.01 --init results/rotate10/best.pt --out_dir results/kmnist
python evaluate.py --ckpt results/kmnist/last.pt --dataset mnist
python evaluate.py --ckpt results/kmnist/last.pt --dataset kmnist
python evaluate.py --plots
```

Сохраняются исходная модель и оценки обоих val в эпохе 0 и после каждой эпохи. Нормализация MNIST train и общая голова остаются неизменными по структуре. Итоговый checkpoint — последняя, десятая эпоха. При прерывании:

```powershell
python train.py --dataset kmnist --aug rotate10 --epochs 10 --lr 0.01 --out_dir results/kmnist --resume results/kmnist/last.pt
```

## Финальный test, отчёт и проверка

После всех обучений и визуального val-анализа:

```powershell
python evaluate.py --report
python evaluate.py --final-test
python evaluate.py --report --check
```

Test-модели и их хеши фиксируются заранее в `results/final_test/selection.json`. После этого новые обучения запрещены. Повторная команда сохраняет уже завершённые test-оценки.

`--report` собирает 11 разделов из реальных артефактов и заполненных наблюдений; ручные изменения `report.md` заменяются. Персональные поля заполните после последней сборки. Если известны мощность и тариф, передайте вместе с `--report` параметры `--power-watts` и `--tariff-per-kwh`; без них денежные суммы пусты.

`--check` проверяет файлы, сплиты, эпохи, LR, resume, исходные веса, полноту train, галереи и метрики. Код возврата 0 — проверки пройдены, 2 — работа не завершена. Сейчас ожидается 2.

Служебные проверки находятся в `results/checks/`: успешно проверены синтаксис, зависимости, признаки, сериализация, отдельные процессы resume и сохранение визуализаций. **Это синтетические проверки кода, а не результаты MNIST.** Их папки `synthetic_*` явно помечены и не используются в отчёте. Автоматическая очистка этих папок отклонена политикой среды. Установочные пакеты остались в `.wheels/` и доступны для установки без сети:

```powershell
python -m pip install --no-index --find-links .wheels -r requirements.txt
```
