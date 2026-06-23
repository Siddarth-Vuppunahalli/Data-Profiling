# Data-Profiling

Python scaffold for database-oriented profiling. The initial foundation profiles
table-shaped data, emits a stable JSON contract, and integrates
`ydata-profiling` as the per-table statistical layer. Custom health and
relationship features from `Database_Profiling_Parameters.pdf` will be added in
separate feature branches.

## Setup

```bash
python -m pip install -e ".[dev]"
```

## Run The Scaffold

The first milestone uses CSV fixtures so the JSON contract and YData integration
can be tested without a live database:

```bash
python -m db_profiler profile \
  --table-csv customers=tests/fixtures/customers.csv \
  --config config/profiling.example.yml \
  --output output/profile.json
```

The CLI already accepts database-oriented options for the next milestone:

```bash
python -m db_profiler profile \
  --database-url postgresql+psycopg://USER:PASS@HOST:5432/DB \
  --schema public \
  --output output/profile.json
```

In this scaffold milestone, at least one `--table-csv TABLE=PATH` input is still
required.

## Test

```bash
python -m pytest
```

## Output Contract

The generated JSON keeps these stable top-level sections:

- `metadata`
- `tables`
- `relationships`
- `database_health`
- `query_generation_hints`

YData-derived metrics live under each table's `ydata_profile` key.
