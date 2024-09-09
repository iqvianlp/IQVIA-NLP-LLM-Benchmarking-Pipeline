import re
import string


def clean_str(
        text: str,
        separator: str = '_',
        include_punct: bool = True,
        include_white_space: bool = True,
        delete_leading: bool = True,
        delete_trailing: bool = True,
        lower: bool = True
) -> str:
    """Cleans potentially problematic characters from strings, by replacing them with a separator.

    Args:
        text: The string to clean.
        separator: The string to replace problematic characters with.
        include_punct: Whether to treat punctuation as problematic and replace them.
        include_white_space: Whether to treat white space as problematic and replace them.
        delete_leading: Whether to delete a leading punctuation; if False, only replaces it.
        delete_trailing: Whether to delete a trailing punctuation; if False, only replaces it.
        lower: Whether to lower-case the result.

    Returns:
        The text string without the original punctuation.

    Raises:
        TypeError: If `text` is not a str.
    """
    if not isinstance(text, str):
        raise TypeError(f'clean_str(text) cannot clean a {type(text)}; text must be string')
    to_replace = ''
    if include_punct:
        to_replace += string.punctuation
    if include_white_space:
        to_replace += string.whitespace
    if to_replace:
        text = text.translate(str.maketrans(dict.fromkeys(to_replace, separator)))
    if delete_leading and text.startswith(separator):
        text = text[1:]
    if delete_trailing and text.endswith(separator):
        text = text[:-1]
    if lower:
        text = text.lower()
    text = re.sub(fr'{separator}+', separator, text)  # remove any duplicate separators
    return text
