# Данные: контракты, форматы, схема истины

## 1. Раскладка на диске

```
data/synthetic/                 # в .gitignore, генерируется целиком
├── static_cloud/
│   └── torus_N400_R2_r0.8_sig0.03_sampling-rejection_seed20260201/
│       ├── points.npy          # (n_points, dim) float64
│       ├── weights.npy         # (n_points, n_points) float64, опционально
│       └── ground_truth.json
├── dynamic_graph/
│   └── cycle_lifecycle_N24_T60_threshold0.5_seed20260201/
│       ├── snapshots.npy       # (T, n_nodes, n_nodes) float64, симметричные
│       └── ground_truth.json
├── series/
│   └── kuramoto_N30_T4000_K1.2_seed20260201/
│       ├── series.npy          # (n_channels, n_timepoints) float64
│       ├── coupling.npy        # (T_schedule, n_channels, n_channels)
│       └── ground_truth.json
├── noisy/
└── null/
└── manifest.json               # реестр всех сгенерированных наборов
```

Имя каталога набора — детерминированная функция от генератора и параметров, включая seed.
Один и тот же конфиг обязан давать то же имя и то же содержимое.

### 1.1. Правило именования

Имя содержит **все** параметры генератора, а не выборочные. Правило неочевидное, поэтому
явно: если параметр не попадает в имя, два набора с разной геометрией (например торы с
разным `R`) получают одинаковое имя и молча перезаписывают друг друга. `naming.py`
отказывается строить имя, если объявленный порядок параметров не покрывает словарь
`params` в точности, — именно эта проверка и не даёт параметру выпасть из имени.

- Порядок параметров объявляется явно рядом с генератором (`register_name_order`), а не
  берётся из порядка ключей `params`: JSON пишется с сортировкой ключей, поэтому порядок
  вставки не переживает round-trip.
- Частые параметры сокращаются ради читаемости: `n_points`/`n_nodes`/`n_channels` → `N`,
  `n_snapshots`/`n_timepoints` → `T`, `noise_sigma` → `sig`. Остальные — под своим именем.
- Числа форматируются как `format(v, f".{naming.float_precision}g")`; строковые значения
  отделяются дефисом (`sampling-rejection`), числовые пишутся слитно (`N400`).
- Seed всегда последний.

## 2. Порядок осей (фиксирован, менять нельзя)

| Объект | Форма | Смысл осей |
|---|---|---|
| Облако точек | `(n_points, dim)` | точки × координаты |
| Матрица весов / расстояний | `(n_nodes, n_nodes)` | симметричная, нули на диагонали для расстояний |
| Динамический граф | `(T, n_nodes, n_nodes)` | время × узел × узел |
| Мультивариативный ряд | `(n_channels, n_timepoints)` | канал × время |
| Расписание связи | `(T_schedule, n_channels, n_channels)` | время × узел × узел |

Ряд хранится как `(n_channels, n_timepoints)` — канал первым. Это отличается от
соглашения фМРТ-модуля, где ряды приходят как `(n_timepoints, n_rois)`; конвертация
делается на границе модулей явной транспозицией, а не «по месту».

## 3. Контракты API

Каждый генератор — функция, принимающая параметры и `rng`, возвращающая датакласс.
Ни один генератор не пишет на диск сам: сохранение — отдельная функция.

```python
@dataclass(frozen=True, eq=False)
class GroundTruth:
    """Machine-readable truth for a generated dataset."""
    format_version: int
    family: str                  # static_cloud | dynamic_graph | series | noisy | null
    generator: str               # torus, kuramoto, cycle_lifecycle, ...
    params: dict                 # фактические аргументы вызова генератора
    seed: int
    coeff_field: int | None = None
    complex: str | None = None                        # "flag" для порогованных графов
    expected_betti: dict[int, int] | None = None      # семейство 1
    betti_curve: dict[int, list[int]] | None = None   # семейство 2, по снимкам
    events: list["Event"] | None = None               # семейство 2
    community_labels: np.ndarray | None = None        # семейство 3, int64
    noise_level: dict | None = None                   # семейство 4
    notes: str = ""

@dataclass(frozen=True)
class Event:
    """A single homological event in a dynamic dataset."""
    snapshot: int      # индекс снимка
    dim: int           # размерность гомологий: 0, 1, 2
    type: str          # ТОЛЬКО "birth" или "death"
    feature_id: str    # устойчивый идентификатор признака
    narrative: str     # человекочитаемое: "две компоненты сливаются"

@dataclass(frozen=True)
class PointCloudDataset:
    points: np.ndarray            # (n_points, dim)
    ground_truth: GroundTruth
    weights: np.ndarray | None = None

@dataclass(frozen=True)
class DynamicGraphDataset:
    snapshots: np.ndarray         # (T, n, n)
    ground_truth: GroundTruth

@dataclass(frozen=True)
class SeriesDataset:
    series: np.ndarray            # (n_channels, n_timepoints)
    coupling: np.ndarray          # (T_schedule, n, n)
    ground_truth: GroundTruth
```

Сигнатура генератора:

```python
def make_torus(n_points: int, R: float, r: float, noise_sigma: float,
               rng: np.random.Generator) -> PointCloudDataset:
```

Правила: `rng` — последний позиционный аргумент, всегда обязателен; глобальное состояние
numpy не используется; функции чистые (одинаковый вход и rng → одинаковый выход).

Датаклассы объявлены с `eq=False`. Сгенерированный `__eq__` сравнивал бы поля через `==`,
что на `np.ndarray` даёт `ValueError: ambiguous truth value`; при `eq=False` сравнение
вырождается в тождество, и тест, сравнивший два разных объекта через `==`, честно падает.
Содержательное сравнение — `dataset_equal` / `ground_truth_equal`, побитовое по массивам.

## 4. Схема ground_truth.json

```json
{
  "format_version": 1,
  "dataset_id": "torus_N400_R2_r0.8_sig0.03_sampling-rejection_seed20260201",
  "family": "static_cloud",
  "generator": "torus",
  "params": {"n_points": 400, "R": 2.0, "r": 0.8, "noise_sigma": 0.03,
             "sampling": "rejection"},
  "seed": 20260201,
  "coeff_field": 2,
  "expected_betti": {"0": 1, "1": 2, "2": 1},
  "notes": "Rejection sampling по элементу площади (R + r cos φ)."
}
```

`dataset_id` в файле есть, но полем датакласса не является: он выводится из
`(generator, params, seed)`. Хранить его ещё и в объекте значило бы позволить имени и
параметрам разъехаться. При чтении он сверяется с именем каталога, а если передан
`float_precision` — то и с пересчитанным заново, что ловит правку `params` руками.

Поля со значением `None` в JSON не пишутся вовсе (а не как `null`): так добавление нового
необязательного поля не меняет побайтово уже сгенерированные файлы.

Для динамического графа:

```json
{
  "format_version": 1,
  "dataset_id": "cycle_lifecycle_N24_T60_threshold0.5_seed20260201",
  "family": "dynamic_graph",
  "generator": "cycle_lifecycle",
  "params": {"n_nodes": 24, "n_snapshots": 60, "threshold": 0.5},
  "seed": 20260201,
  "coeff_field": 2,
  "complex": "flag",
  "betti_curve": {"0": [1, 1, "..."], "1": [0, 0, "..."], "2": ["..."]},
  "events": [
    {"snapshot": 12, "dim": 1, "type": "birth", "feature_id": "h1_a",
     "narrative": "путь замыкается в цикл"},
    {"snapshot": 27, "dim": 1, "type": "birth", "feature_id": "h1_b",
     "narrative": "хорда делит цикл на два"},
    {"snapshot": 41, "dim": 1, "type": "death", "feature_id": "h1_b",
     "narrative": "удалено ребро"}
  ]
}
```

**Ключевое требование к согласованности:** `events` и `betti_curve` — два независимых
источника. `events` отражает намерение планировщика, `betti_curve` считается независимой
эталонной реализацией из `reference/`. Тест обязан проверить, что скачки `betti_curve`
в точности соответствуют списку событий. Расхождение означает баг в генераторе, а не
повод подправить лог.

## 5. manifest.json

Реестр всех наборов: `dataset_id`, семейство, генератор, seed, путь относительно
`output_root`, `params`, sha256 массивов, версия пакета, дата генерации. Служит для
воспроизводимости и для быстрого обнаружения того, что данные пересобраны с другими
параметрами. Структура: `{"format_version": 1, "datasets": [ ... ]}`, записи
отсортированы по `dataset_id`.

- sha256 считается **по байтам файла `.npy`**, а не по `ndarray.tobytes()`: результат
  проверяем обычным `shasum -a 256`, и в хеш попадает заголовок, поэтому изменившиеся
  dtype или форма не спрячутся за одинаковой полезной нагрузкой.
- Сам манифест побайтово невоспроизводим: в нём записано время генерации. Требование
  «пересборка даёт те же хеши» проверяется на записи без `generated_at` и
  `package_version` (`entry_payload`).

## 6. Правила сериализации

- Массивы — `.npy` (не pickle), float64.
- JSON — UTF-8, `ensure_ascii=False`, отсортированные ключи, отступ 2.
- Ключи размерностей гомологий в JSON — строки ("0", "1", "2"), при чтении приводятся
  к int. Это ограничение JSON, а не выбор дизайна; конвертацию делать в одном месте.
- `expected_betti` и `betti_curve` перечисляют **все** размерности от 0 до
  `max_homology_dim`, включая нулевые: отсутствующий ключ неотличим от забытого.
- `community_labels` хранится внутри `ground_truth.json` (отдельного `.npy` для него нет)
  и фиксирован как `int64`, иначе round-trip молча менял бы dtype.
- Ничего не дописывать в существующий каталог набора: пересборка — только целиком.
- Тип набора при чтении определяется по имени файла массива (`points.npy`,
  `snapshots.npy`, `series.npy`), а не по `family`: у зашумлённого облака и
  зашумлённого динамического графа семейство одно и то же — `noisy`.
