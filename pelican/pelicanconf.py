#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Masato Hagiwara'
SITEURL = ''
SITENAME = 'Real-World Natural Language Processing'
SITETITLE = 'Real-World Natural Language Processing'
# SITESUBTITLE = 'NLP recipes and best practices for the Manning book "Real-World NLP"'
SITELOGO = 'http://masatohagiwara.net/img/profile.jpg'

# SITEDESCRIPTION = 'Foo Bar\'s Thoughts and Writings'
# SITELOGO = SITEURL + '/images/profile.png'
# FAVICON = SITEURL + '/images/favicon.ico'
#
# BROWSER_COLOR = '#333'
# ROBOTS = 'index, follow'
#
# CC_LICENSE = {
#     'name': 'Creative Commons Attribution-ShareAlike',
#     'version': '4.0',
#     'slug': 'by-sa'
# }

COPYRIGHT_YEAR = 2018

# EXTRA_PATH_METADATA = {
#     'extra/custom.css': {'path': 'static/custom.css'},
# }
# CUSTOM_CSS = 'static/custom.css'
#
# MAIN_MENU = True
#
# ADD_THIS_ID = 'ra-77hh6723hhjd'
# DISQUS_SITENAME = 'yoursite'
GOOGLE_ANALYTICS = 'UA-175204-17'
# GOOGLE_TAG_MANAGER = 'GTM-ABCDEF'
# STATUSCAKE = { 'trackid': 'your-id', 'days': 7, 'design': 6, 'rumid': 1234 }
#
# # Enable i18n plugin.
# PLUGINS = ['i18n_subsites']
# # Enable Jinja2 i18n extension used to parse translations.
# JINJA_EXTENSIONS = ['jinja2.ext.i18n']
#
# # Translate to German.
# DEFAULT_LANG = 'de'
# OG_LOCALE = 'de_DE'
# LOCALE = 'de_DE'
#
# # Default theme language.
I18N_TEMPLATES_LANG = 'en'

PATH = 'content'

TIMEZONE = 'America/New_York'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (('Home', 'http://www.realworldnlpbook.com/'),
         ('Masato Hagiwara', 'http://masatohagiwara.net/'),
         ('Manning Publications', 'https://www.manning.com/'),)


# Social widget
SOCIAL = (('envelope', 'mailto: hagisan@gmail.com'),
          ('twitter', 'https://twitter.com/mhagiwara'),
          ('github', 'https://github.com/mhagiwara/realworldnlp'),)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True

THEME = './Flex'

PYGMENTS_STYLE = 'monokai'
