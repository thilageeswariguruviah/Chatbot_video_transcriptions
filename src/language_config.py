"""
Language Configuration for Chatbot

This file contains language mapping configurations that determine how the chatbot
responds to different detected languages.

Modify the LANGUAGE_MAPPING dictionary to change the behavior:
- Key: detected language from user input
- Value: language to use in chatbot response
"""

# Current configuration: Each language responds in its own language
LANGUAGE_MAPPING = {
    # Each Indian language responds in its own language
    'telugu_english': 'telugu_english',    # Telugu questions → Telugu responses
    'hindi_english': 'hindi_english',      # Hindi questions → Hindi responses
    'kannada_english': 'kannada_english',  # Kannada questions → Kannada responses
    'malayalam_english': 'malayalam_english', # Malayalam questions → Malayalam responses
    'tamil_english': 'tamil_english',      # Tamil questions → Tamil responses
    
    # Keep English as English
    'english': 'english',
    
    # Other languages
    'spanish': 'spanish',
    'french': 'french',
    'german': 'german',
    'italian': 'italian',
    'portuguese': 'portuguese',
}

# Alternative configurations (uncomment to use):

# Configuration 1: Keep each language in its own language
# LANGUAGE_MAPPING = {
#     'telugu_english': 'telugu_english',
#     'hindi_english': 'hindi_english',
#     'kannada_english': 'kannada_english',
#     'malayalam_english': 'malayalam_english',
#     'tamil_english': 'tamil_english',
#     'english': 'english',
# }

# Configuration 2: Map everything to English
# LANGUAGE_MAPPING = {
#     'telugu_english': 'english',
#     'hindi_english': 'english',
#     'kannada_english': 'english',
#     'malayalam_english': 'english',
#     'tamil_english': 'english',
#     'english': 'english',
# }

# Configuration 3: Regional mapping
# LANGUAGE_MAPPING = {
#     'telugu_english': 'telugu_english',    # Andhra Pradesh/Telangana
#     'tamil_english': 'tamil_english',      # Tamil Nadu  
#     'kannada_english': 'kannada_english',  # Karnataka
#     'malayalam_english': 'malayalam_english', # Kerala
#     'hindi_english': 'hindi_english',      # North India
#     'english': 'english',
# }

# Configuration 4: Custom user-specific mapping
# LANGUAGE_MAPPING = {
#     'telugu_english': 'hindi_english',     # Telugu users get Hindi responses
#     'hindi_english': 'tamil_english',      # Hindi users get Tamil responses
#     'kannada_english': 'english',          # Kannada users get English responses
#     'malayalam_english': 'malayalam_english', # Malayalam users get Malayalam responses
#     'tamil_english': 'tamil_english',      # Tamil users get Tamil responses
#     'english': 'english',
# }