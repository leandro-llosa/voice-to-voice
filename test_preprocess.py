"""Sanity checks for app.detect_language and app.preprocess.

Plain script, no test framework needed. Run with:

    .venv/bin/python test_preprocess.py
"""

import re
import sys

from app import CODE_PLACEHOLDER, detect_language, preprocess

EN_MD = """# Weekly Report

**Update:** we ran `pytest` across the service and rebuilt the Docker image.
Full log at https://example.com/ci/logs/very/long/path/runner.html [1].

- Clone the repo
- Run the migrations
- Deploy to staging

```python
def hello():
    print("hi")
```

That is all for this week.
"""

ES_MD = """# Resumen semanal

**Importante:** revisamos `config.yaml` y el identificador `some_var_name` antes del despliegue.
Más detalles en https://ejemplo.com/documentacion/ruta/muy/larga/pagina.html [2].

- Revisar los registros
- Actualizar las dependencias
- Escribir la documentación

```bash
export SOME_VAR_NAME=1
```

¿Tienes alguna pregunta sobre esto?
"""


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"PASS {name}")
        return
    message = f"FAIL {name}: {detail}" if detail else f"FAIL {name}"
    print(message)
    sys.exit(1)


def main() -> None:
    en = preprocess(EN_MD, "en")
    check("en: fenced code removed", "```" not in en)
    check("en: bold markers removed", "**" not in en)
    check("en: url removed", "http" not in en)
    check("en: citation removed", "[1]" not in en)
    check("en: code placeholder present", CODE_PLACEHOLDER["en"] in en)
    check("en: inline code kept (pytest)", "pytest" in en)
    check("en: inline code kept (Docker)", "Docker" in en)
    check("en: bullet became sentence", "Clone the repo." in en)
    check("en: prose preserved", "That is all for this week." in en)

    es = preprocess(ES_MD, "es")
    check("es: fenced code removed", "```" not in es)
    check("es: url removed", "http" not in es)
    check("es: code placeholder present", CODE_PLACEHOLDER["es"] in es)
    check("es: filename kept", "config.yaml" in es)
    check("es: snake_case kept", "some_var_name" in es)
    check("es: citation removed", "[2]" not in es)
    check("es: question kept", "¿Tienes alguna pregunta sobre esto?" in es)

    check("detect: en markdown", detect_language(EN_MD) == "en")
    check("detect: es markdown", detect_language(ES_MD) == "es")
    check(
        "detect: plain english sentence",
        detect_language("Hello there. This is a plain English reply.") == "en",
    )
    check(
        "detect: plain spanish sentence",
        detect_language("Esto es una respuesta en español muy natural.") == "es",
    )

    plain = "Hello world, this is a simple test."
    out = preprocess(plain, "en")
    check(
        "plain text: words unchanged",
        re.findall(r"\w+", plain) == re.findall(r"\w+", out),
        f"got {out!r}",
    )

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
