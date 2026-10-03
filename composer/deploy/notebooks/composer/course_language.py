"""Display the native notebook language selector without changing course files."""
import json
from pathlib import Path
import sys
import uuid


def show_language_selector():
    from IPython.display import HTML, Javascript, display
    if sys.platform == 'emscripten':
        display(HTML('<p>Use the language control above this browser lesson.</p>'))
        return
    graph = json.loads(Path(__file__).with_name('translation_web.json').read_text())
    labels = json.loads(Path(__file__).with_name('course_language_text.json').read_text())
    root = Path(__file__).resolve().parent.parent
    language = json.loads((root / graph['entrypoint']).read_text()).get('metadata', {}).get('course_language', graph['languages'][0])
    identifier = 'course-language-' + uuid.uuid4().hex
    display(HTML(f'<div id="{identifier}" class="course-language-control" role="group" aria-label="{labels[language]["language"]}">{labels[language]["loading"]}</div>'))
    source = Path(__file__).with_name('course_language.js').read_text()
    display(Javascript(source.replace('__COURSE_LANGUAGE_CONFIG__', json.dumps({'id': identifier, 'entrypoints': graph['selector_entrypoints'], 'labels': labels, 'language': language}))))
