"""Character-specific FastAPI extensions for `harness.web`.

Drop a module named `<character_name>.py` here exporting
`build_router(character, adapter) -> APIRouter`; `build_character_app`
discovers and mounts it automatically when serving that character. See
the TFR explainer extension (harness-3jz1.4) for the first concrete
example.
"""
