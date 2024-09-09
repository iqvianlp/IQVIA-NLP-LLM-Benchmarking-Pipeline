from typing import Any


PROTECTED_SECTIONS = ['example', 'cue']


def get_prompt(task, template, inputt, example_map, test_example_id):
    _clean_fillers(task)
    _filter_sections(task, template)
    return _build_prompt(task, template, inputt, example_map, test_example_id)


def _clean_fillers(task):
    task['sections'] = {k: v.strip() for k, v in task['sections'].items()}


def _filter_sections(task, template):
    sections_to_keep = PROTECTED_SECTIONS + list(task['sections'].keys())
    template['sections'] = [x for x in template['sections'] if x['name'] in sections_to_keep]


def _build_prompt(task, template, inputt, example_map, test_example_id):
    """
    Builds a prompt template by joining the sections that the template should contain.

    If a section is an example, retrieve the best example with the example selection algorithm and feed it into the
    example template of the task.

    Always add a cue section, using the same example template above, populated with the actual query to the model.

    Preempts a section with a header, if any.
    """
    result = ''
    train_example_idx = 0
    for s in template['sections']:
        if 'header' in s:
            result += f'{s["header"]}\n'
        if s['name'] == 'example':
            result += _get_example(task, inputt, example_map, test_example_id, train_example_idx)
            train_example_idx += 1
        elif s['name'] == 'cue':
            result += _get_example(task, inputt, example_map, test_example_id, train_example_idx, is_cue=True)
        else:
            result += f'{task["sections"][s["name"]]}\n'
    return result.strip()


def _get_example(task, inputt, example_map, test_example_id, example_idx, is_cue=False):
    if is_cue:
        fillers = inputt + ['']  # a cue is effectively an example section with an empty answer
        return task['example'].format(*fillers).strip() + '\n'
    else:
        sims = example_map[task['dataset']][test_example_id]['sims']
        inputs = sims[example_idx]['input']
        output = sims[example_idx]['output']
        fillers = inputs + [output]
        slots = task['example'].count('{}')
        if slots != len(fillers):
            print(f'No example added for example {task["dataset"]}/{test_example_id}; expected {slots} slots as per '
                  f'example template, but {len(fillers)} fillers provided.')
            return ''
        return 'Example:\n' + task['example'].format(*fillers).strip() + '\n'


def get_chat_prompt(
        task: dict[str, Any],
        template: dict[str, dict],
        inputt: list[str],
        example_map: dict,
        test_example_id: str,
        example_output_placeholder: str,
        use_system_role: bool = True
) -> list[dict[str, str]]:
    _clean_fillers(task)
    _filter_sections(task, template)

    result = list()
    train_example_idx = 0
    newline = '\n'
    for s in template['sections']:
        if s['name'] == 'example':
            example_output = example_map[task['dataset']][test_example_id]['sims'][train_example_idx]['output']
            result = add_content_by_user(
                result,
                {
                    "role": "user",
                    "content":
                        f'{s["header"] + newline if "header" in s else ""}'
                        f'{_get_example_for_chat(task, inputt, example_map, test_example_id, train_example_idx)}'
                }
            )
            result = add_content_by_user(
                result,
                {
                    "role": "assistant",
                    "content": str(example_output) if str(example_output) != '' else str(example_output_placeholder)
                }
            )
            train_example_idx += 1
        elif s['name'] == 'cue':
            result = add_content_by_user(
                result,
                {
                    "role": "user",
                    "content": _get_example_for_chat(
                        task, inputt, example_map, test_example_id, train_example_idx, is_cue=True
                    )
                }
            )
        else:
            result = add_content_by_user(
                result,
                {
                    "role": "system" if use_system_role and s['name'] in ['short-instructions', 'long-instructions']
                    else "user",
                    "content": f'{s["header"] + newline if "header" in s else ""}{task["sections"][s["name"]]}'
                }
            )
    return result


def _get_example_for_chat(
        task: dict,
        inputt: list[str],
        example_map: dict,
        test_example_id: str,
        example_idx: int,
        is_cue: bool = False
) -> str:
    fillers = inputt if is_cue else example_map[task['dataset']][test_example_id]['sims'][example_idx]['input']
    fillers += ['']
    return task['example'].format(*fillers).strip() + '\n'


def add_content_by_user(existing_messages: list, new_message: dict) -> list:
    if len(existing_messages) < 1 or existing_messages[-1]["role"] != new_message["role"]:
        existing_messages.append(new_message)
    else:
        existing_messages[-1]["content"] += '\n' + new_message["content"]
    return existing_messages
