"""
helper.py

Helper functions used across the Vocalytics project.
"""

import os


def create_directory(directory):
    """
    Create a directory if it does not exist.
    """

    if not os.path.exists(directory):
        os.makedirs(directory)


def get_file_extension(file_name):
    """
    Return file extension.
    """

    return os.path.splitext(file_name)[1].lower()


def format_confidence(confidence):
    """
    Format confidence value.
    """

    return f"{confidence:.2f}%"