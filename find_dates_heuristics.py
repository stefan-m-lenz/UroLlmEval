import Levenshtein
import re

DATE_PATTERN=r"\b(\d{4}|\d{1,2}/\d{4}|\d{1,2}/\d{2}|\d{1,2}\.\d{1,2}\.\d{4})\b"

def fuzzy_find(text_a, text_b):
    """
    Find the position of the maximum overlap of text_b in text_a
    while being resilient to small errors using Levenshtein distance.
    """
    len_a = len(text_a)
    len_b = len(text_b)

    if len_a < len_b:
        raise ValueError("text_a must be at least as long as text_b")

    min_distance = float('inf')
    best_position = -1

    for i in range(len_a - len_b + 1):
        substring = text_a[i:i + len_b]
        distance = Levenshtein.distance(substring, text_b)
        if distance < min_distance:
            min_distance = distance
            best_position = i

    start = best_position
    end = best_position + len_b

    # Extend the match to the left and right
    start = best_position
    end = best_position + len_b

    # Attempt to shrink the match
    # Check shrinking from the end
    while end > start + 1 and Levenshtein.distance(text_a[start:end-1], text_b) < min_distance:
        end -= 1
        min_distance = Levenshtein.distance(text_a[start:end], text_b)

    # Check shrinking from the start
    while start < end - 1 and Levenshtein.distance(text_a[start+1:end], text_b) < min_distance:
        start += 1
        min_distance = Levenshtein.distance(text_a[start:end], text_b)

    # Extend to the right
    while end < len_a and Levenshtein.distance(text_a[start:end+1], text_b) <= min_distance:
        end += 1

    # Extend to the left
    while start > 0 and Levenshtein.distance(text_a[start-1:end], text_b) <= min_distance:
        start -= 1

    # Remove leading and trailing whitespace from the matched substring
    trimmed_start = start
    trimmed_end = end
    while trimmed_start < trimmed_end and text_a[trimmed_start].isspace():
        trimmed_start += 1
    while trimmed_end > trimmed_start and text_a[trimmed_end - 1].isspace():
        trimmed_end -= 1

    trimmed_match = text_a[trimmed_start:trimmed_end]

    return trimmed_start, trimmed_match


import re

DATE_PATTERN = r"\b(\d{4}|\d{1,2}/\d{4}|\d{1,2}/\d{2}|\d{1,2}\.\d{1,2}\.\d{4})\b"

def find_most_likely_dates(text_a, text_b_start, text_b_end, max_line_dist=0):
    """
    Find the most likely dates associated with text_b in text_a, based on line distance.

    Args:
    text_a (str): The larger text containing dates and other information.
    text_b_start (int): The start position of text_b in text_a.
    text_b_end (int): The end position of text_b in text_a.
    max_line_dist (int): The maximum line distance from text_b to consider for a date match.

    Returns:
    list of str: The most likely dates in the neighbouring lines sorted by proximity.
    """
    lines = text_a.split('\n')
    line_indices = []  # This will store the line indices of text_b

    # Find out which lines text_b spans
    current_length = 0
    for i, line in enumerate(lines):
        next_length = current_length + len(line) + 1  # +1 for the newline character
        if current_length <= text_b_start < next_length:
            line_indices.append(i)
        if current_length < text_b_end <= next_length:
            line_indices.append(i)
            break
        current_length = next_length

    date_matches = [] # array of tuples, each tuple containing two for the date strings and the distances

    # Now, iterate over all lines and find dates, calculating line distance
    for i, line in enumerate(lines):
        line_dates = re.finditer(DATE_PATTERN, line)
        for match in line_dates:
            line_distance = min(abs(i - idx) for idx in line_indices)
            if line_distance <= max_line_dist:
                date_matches.append((match.group(), line_distance))

    # Sort the matches by line distance
    date_matches.sort(key=lambda x: x[1])

    # Return only the date strings
    return [date for date, _ in date_matches]


def fuzzy_find_dates(text, event_text, max_line_dist):
    text_b_start, text_b_fuzzy_match = fuzzy_find(text_a=text, text_b=event_text)
    text_b_end = text_b_start + len(text_b_fuzzy_match)
    return find_most_likely_dates(text, text_b_start, text_b_end, max_line_dist)
