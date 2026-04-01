# Contributing

## Development Setup

Install the project in editable mode:

```bash
pip install -e .
```

The repository uses Python `>=3.12` and keeps runtime settings in [configs/settings.py](configs/settings.py).

## Working Style

- Keep preprocessing stages scriptable from the command line.
- Prefer class-based orchestration for large stages.
- Preserve backward compatibility for existing CLI flags and shard schemas unless there is a deliberate migration.
- Do not silently change metadata field semantics.

## Testing

Run focused preprocessing tests before opening a change:

```bash
python -m unittest tests.test_face_detection tests.test_face_detection_split tests.test_face_detection_crop
python -m compileall preprocessing tests
```

When a change affects another stage, add or update tests in `tests/` for that stage as part of the same change.

## Documentation

Update the relevant docs when you change:

- CLI flags
- metadata schema
- shard contents
- stage ordering
- default output locations

The main documentation entry points are:

- [README.md](README.md)
- [docs/preprocessing-pipeline.md](docs/preprocessing-pipeline.md)
- [docs/data-contracts.md](docs/data-contracts.md)
- [docs/architecture.md](docs/architecture.md)

## Commit Guidance

Use short, descriptive commit messages with a clear scope, for example:

```text
feat(preprocessing): separate align canvas from final face crop size
fix(face-detection): isolate split-specific audit and shard output
docs: refresh preprocessing pipeline documentation
```
