def file_length(path):

    """Simply count the lines in a text file."""

    with open(path) as f:
        for i, l in enumerate(f):  # read through the lines in the file (i counts and l stores each line)
            pass
    return i + 1