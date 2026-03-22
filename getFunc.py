"""
getFunc.py — regex-based C context extractor for LLM prompts.

Usage:  python getFunc.py [FunctionName]
Default target: Input_Poll
"""

import os
import re
import sys

GENGIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gengin")

_SKIP = {
    'if', 'for', 'while', 'switch', 'do', 'else', 'return',
    'sizeof', 'typeof', '__typeof__', 'alignof', 'offsetof',
}

# Matches function definitions:  [qualifiers] rettype funcname(params) {
_FUNC_RE = re.compile(
    r'^((?:(?:static|inline|extern|const|unsigned|signed|void|struct)\s+)*'
    r'[\w\s\*]+?)\s+(\w+)\s*\(([^;{]*?)\)\s*\{',
    re.MULTILINE,
)

# Matches typedef struct { ... } Name;  or  struct Name { ... };
_STRUCT_RE = re.compile(r'(typedef\s+)?struct\s+(\w*)\s*\{', re.MULTILINE)


def _read_sources(base_dir):
    sources = {}
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(('.c', '.h')):
                path = os.path.join(root, f)
                with open(path, errors='replace') as fh:
                    sources[path] = fh.read()
    return sources


def _strip_comments(text):
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'//[^\n]*', '', text)
    return text


def _extract_block(text, brace_pos):
    """Return the brace-balanced block starting at brace_pos (must be '{')."""
    depth = 0
    for i in range(brace_pos, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[brace_pos:i + 1]
    return text[brace_pos:]


def find_functions(sources):
    """Return {name: {'sig', 'body', 'full', 'file'}} for all function defs."""
    results = {}
    for filepath, raw in sources.items():
        content = _strip_comments(raw)
        for m in _FUNC_RE.finditer(content):
            name = m.group(2)
            if name in _SKIP:
                continue
            brace_pos = m.start() + m.group(0).rfind('{')
            body = _extract_block(content, brace_pos)
            sig = m.group(0)[:m.group(0).rfind('{')].strip()
            results[name] = {
                'sig': sig,
                'body': body,
                'full': sig + '\n' + body,
                'file': os.path.relpath(filepath, GENGIN),
            }
    return results


def find_structs(sources):
    """Return {name: {'full', 'file'}} for all struct/typedef-struct defs."""
    results = {}
    for filepath, raw in sources.items():
        content = _strip_comments(raw)
        for m in _STRUCT_RE.finditer(content):
            brace_pos = m.start() + m.group(0).rfind('{')
            body = _extract_block(content, brace_pos)
            end_pos = brace_pos + len(body)
            after = content[end_pos:end_pos + 64]
            td_match = re.match(r'\s*(\w+)\s*;', after)
            struct_tag = m.group(2) or ''
            typedef_name = td_match.group(1) if (m.group(1) and td_match) else None
            key = typedef_name or struct_tag
            if not key:
                continue
            full = content[m.start():end_pos]
            if typedef_name:
                full += td_match.group(0)
            entry = {'full': full.strip(), 'file': os.path.relpath(filepath, GENGIN)}
            results[key] = entry
            if struct_tag and struct_tag != key:
                results[struct_tag] = entry
    return results


def run_bench():
    """
    Run `make bench` in the gengin directory and return the parsed JSON results.
    Returns a dict on success, or None if the build/run failed.
    """
    import subprocess
    import json

    result_path = os.path.join(GENGIN, 'bench_results.json')

    proc = subprocess.run(
        ['make', 'bench'],
        cwd=GENGIN,
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"run_bench: make bench failed:\n{proc.stderr.strip()}")
        return None

    if not os.path.exists(result_path):
        print(f"run_bench: bench_results.json not found after make bench.")
        return None

    with open(result_path) as fh:
        data = json.load(fh)

    return data


def restore_file(filepath):
    """Restore a single file to its last committed state via git checkout."""
    import subprocess
    result = subprocess.run(
        ['git', 'checkout', 'HEAD', '--', filepath],
        cwd=GENGIN,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"restore_file failed: {result.stderr.strip()}")
        return False
    print(f"Restored {os.path.relpath(filepath, GENGIN)}")
    return True


def restore_function(func_name, functions):
    """
    Restore the source file that contains func_name to its last committed state.
    Uses `git checkout HEAD -- <file>` so any edits made by replace_function
    (or anything else) are discarded.
    """
    if func_name not in functions:
        print(f"restore_function: '{func_name}' not found.")
        return False
    rel = functions[func_name]['file']
    filepath = os.path.join(GENGIN, rel)
    return restore_file(filepath)


def restore_all():
    """Restore the entire gengin directory to its last committed state."""
    import subprocess
    result = subprocess.run(
        ['git', 'checkout', 'HEAD', '--', '.'],
        cwd=GENGIN,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"restore_all failed: {result.stderr.strip()}")
        return False
    print("Restored all files in gengin/ to HEAD.")
    return True


def list_functions(functions):
    """
    Return a sorted list of dicts with basic info about every function found:
      [{'name', 'file', 'sig'}, ...]
    Useful as an index before deciding what to extract or replace.
    """
    return sorted(
        [{'name': name, 'file': data['file'], 'sig': data['sig']}
         for name, data in functions.items()],
        key=lambda x: (x['file'], x['name']),
    )


def _called_functions(body):
    return set(re.findall(r'\b(\w+)\s*\(', body)) - _SKIP


def _used_types(text):
    # Heuristic: C type names are CamelCase or ALL_CAPS identifiers
    return set(re.findall(r'\b([A-Z][A-Za-z0-9_]*)\b', text))


def replace_function(func_name, new_definition, functions, sources):
    """
    Replace the definition of func_name in its source file with new_definition.

    new_definition should be the full replacement text (signature + body).
    Returns True on success, False if the function wasn't found or the original
    text couldn't be located in the file.
    """
    if func_name not in functions:
        print(f"replace_function: '{func_name}' not found in parsed functions.")
        return False

    info = functions[func_name]
    filepath = None
    for path in sources:
        if os.path.relpath(path, GENGIN) == info['file']:
            filepath = path
            break

    if filepath is None:
        print(f"replace_function: source file for '{func_name}' not found.")
        return False

    original_text = sources[filepath]
    old_full = info['full']

    # Find the original (non-comment-stripped) span by matching the signature
    # then walking braces, so we preserve original formatting everywhere else.
    sig_pattern = re.escape(info['sig'].split('(')[0].strip().split()[-1])
    # Locate the function via its name followed by '('
    func_start_re = re.compile(
        r'(?:^|\n)((?:(?:static|inline|extern|const|unsigned|signed|void|struct)\s+)*'
        r'[\w\s\*]+?\s+)?' + sig_pattern + r'\s*\([^;]*?\)\s*\{',
        re.MULTILINE,
    )
    match = func_start_re.search(original_text)
    if match is None:
        print(f"replace_function: could not locate '{func_name}' in {filepath}.")
        return False

    brace_pos = match.start() + match.group(0).rfind('{')
    old_block = _extract_block(original_text, brace_pos)
    old_span_start = match.start() if original_text[match.start()] == '\n' else match.start()
    old_span_end = brace_pos + len(old_block)

    new_text = original_text[:old_span_start] + '\n' + new_definition.strip() + '\n' + original_text[old_span_end:]

    with open(filepath, 'w') as fh:
        fh.write(new_text)

    print(f"replace_function: replaced '{func_name}' in {os.path.relpath(filepath, GENGIN)}")
    return True


def show_context(target_func, functions, structs, returnString = False):
    if target_func not in functions:
        print(f"Function '{target_func}' not found in codebase.")
        return None

    target = functions[target_func]
    all_types = set()
    lines = []

    lines.append(f"// {'=' * 60}")
    lines.append(f"// TARGET: {target_func}  [{target['file']}]")
    lines.append(f"// {'=' * 60}")
    lines.append(target['full'])
    all_types |= _used_types(target['full'])

    callees = {n: functions[n] for n in _called_functions(target['body']) if n in functions}
    if callees:
        lines.append(f"\n// {'=' * 60}")
        lines.append(f"// CALLED FUNCTIONS (1 level deep)")
        lines.append(f"// {'=' * 60}")
        for fname, fdata in sorted(callees.items()):
            lines.append(f"\n// --- {fname}  [{fdata['file']}] ---")
            lines.append(fdata['full'])
            all_types |= _used_types(fdata['full'])

    found_structs = {t: structs[t] for t in all_types if t in structs}
    if found_structs:
        lines.append(f"\n// {'=' * 60}")
        lines.append(f"// USED STRUCTS / TYPES")
        lines.append(f"// {'=' * 60}")
        for tname, tdata in sorted(found_structs.items()):
            lines.append(f"\n// --- {tname}  [{tdata['file']}] ---")
            lines.append(tdata['full'])

    output = '\n'.join(lines)
    if returnString:
        return output
    print(output)


if __name__ == '__main__':
    # Usage:
    #   python getFunc.py [FunctionName]
    #   python getFunc.py [FunctionName] --replace new_impl.c
    args = sys.argv[1:]
    target = args[0] if args and not args[0].startswith('--') else 'RayTraceRowFunc'

    sources = _read_sources(GENGIN)
    functions = find_functions(sources)
    structs = find_structs(sources)

    if '--list' in args:
        for entry in list_functions(functions):
            print(f"{entry['file']:<45}  {entry['name']}")
    elif '--bench' in args:
        import json
        data = run_bench()
        if data:
            print(json.dumps(data, indent=2))
    elif '--restore-all' in args:
        restore_all()
    elif '--restore' in args:
        restore_function(target, functions)
    elif '--replace' in args:
        idx = args.index('--replace')
        if idx + 1 >= len(args):
            print("Usage: python getFunc.py FunctionName --replace new_impl.c")
            sys.exit(1)
        new_file = args[idx + 1]
        with open(new_file) as fh:
            new_def = fh.read()
        replace_function(target, new_def, functions, sources)
    else:
        show_context(target, functions, structs)
